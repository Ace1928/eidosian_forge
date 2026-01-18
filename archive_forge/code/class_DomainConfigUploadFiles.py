import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
class DomainConfigUploadFiles(object):

    def __init__(self, domain_config_finder=_domain_config_finder):
        super(DomainConfigUploadFiles, self).__init__()
        self.load_backends()
        self._domain_config_finder = domain_config_finder

    def load_backends(self):
        drivers = backends.load_backends()
        self.resource_manager = drivers['resource_api']
        self.domain_config_manager = drivers['domain_config_api']

    def valid_options(self):
        """Validate the options, returning True if they are indeed valid.

        It would be nice to use the argparse automated checking for this
        validation, but the only way I can see doing that is to make the
        default (i.e. if no optional parameters are specified) to upload
        all configuration files - and that sounds too dangerous as a
        default. So we use it in a slightly unconventional way, where all
        parameters are optional, but you must specify at least one.

        """
        if CONF.command.all is False and CONF.command.domain_name is None:
            print(_('At least one option must be provided, use either --all or --domain-name'))
            return False
        if CONF.command.all is True and CONF.command.domain_name is not None:
            print(_('The --all option cannot be used with the --domain-name option'))
            return False
        return True

    def _upload_config_to_database(self, file_name, domain_name):
        """Upload a single config file to the database.

        :param file_name: the file containing the config options
        :param domain_name: the domain name
        :returns: a boolean indicating if the upload succeeded

        """
        try:
            domain_ref = self.resource_manager.get_domain_by_name(domain_name)
        except exception.DomainNotFound:
            print(_('Invalid domain name: %(domain)s found in config file name: %(file)s - ignoring this file.') % {'domain': domain_name, 'file': file_name})
            return False
        if self.domain_config_manager.get_config_with_sensitive_info(domain_ref['id']):
            print(_('Domain: %(domain)s already has a configuration defined - ignoring file: %(file)s.') % {'domain': domain_name, 'file': file_name})
            return False
        sections = {}
        try:
            parser = cfg.ConfigParser(file_name, sections)
            parser.parse()
        except Exception:
            print(_('Error parsing configuration file for domain: %(domain)s, file: %(file)s.') % {'domain': domain_name, 'file': file_name})
            return False
        try:
            for group in sections:
                for option in sections[group]:
                    sections[group][option] = sections[group][option][0]
            self.domain_config_manager.create_config(domain_ref['id'], sections)
            return True
        except Exception as e:
            msg = 'Error processing config file for domain: %(domain_name)s, file: %(filename)s, error: %(error)s'
            LOG.error(msg, {'domain_name': domain_name, 'filename': file_name, 'error': e}, exc_info=True)
            return False

    def read_domain_configs_from_files(self):
        """Read configs from file(s) and load into database.

        The command line parameters have already been parsed and the CONF
        command option will have been set. It is either set to the name of an
        explicit domain, or it's None to indicate that we want all domain
        config files.

        """
        domain_name = CONF.command.domain_name
        conf_dir = CONF.identity.domain_config_dir
        if not os.path.exists(conf_dir):
            print(_('Unable to locate domain config directory: %s') % conf_dir)
            raise ValueError
        if domain_name:
            fname = DOMAIN_CONF_FHEAD + domain_name + DOMAIN_CONF_FTAIL
            if not self._upload_config_to_database(os.path.join(conf_dir, fname), domain_name):
                return False
            return True
        success_cnt = 0
        failure_cnt = 0
        for filename, domain_name in self._domain_config_finder(conf_dir):
            if self._upload_config_to_database(filename, domain_name):
                success_cnt += 1
                LOG.info('Successfully uploaded domain config %r', filename)
            else:
                failure_cnt += 1
        if success_cnt == 0:
            LOG.warning('No domain configs uploaded from %r', conf_dir)
        if failure_cnt:
            return False
        return True

    def run(self):
        try:
            self.resource_manager.list_domains(driver_hints.Hints())
        except Exception:
            print(_('Unable to access the keystone database, please check it is configured correctly.'))
            raise
        if not self.valid_options():
            return 1
        if not self.read_domain_configs_from_files():
            return 1