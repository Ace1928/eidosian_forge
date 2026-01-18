import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
class DomainConfigs(provider_api.ProviderAPIMixin, dict):
    """Discover, store and provide access to domain specific configs.

    The setup_domain_drivers() call will be made via the wrapper from
    the first call to any driver function handled by this manager.

    Domain specific configurations are only supported for the identity backend
    and the individual configurations are either specified in the resource
    database or in individual domain configuration files, depending on the
    setting of the 'domain_configurations_from_database' config option.

    The result will be that for each domain with a specific configuration,
    this class will hold a reference to a ConfigOpts and driver object that
    the identity manager and driver can use.

    """
    configured = False
    driver = None
    _any_sql = False
    lock = threading.Lock()

    def _load_driver(self, domain_config):
        return manager.load_driver(Manager.driver_namespace, domain_config['cfg'].identity.driver, domain_config['cfg'])

    def _load_config_from_file(self, resource_api, file_list, domain_name):

        def _assert_no_more_than_one_sql_driver(new_config, config_file):
            """Ensure there is no more than one sql driver.

            Check to see if the addition of the driver in this new config
            would cause there to be more than one sql driver.

            """
            if new_config['driver'].is_sql and (self.driver.is_sql or self._any_sql):
                raise exception.MultipleSQLDriversInConfig(source=config_file)
            self._any_sql = self._any_sql or new_config['driver'].is_sql
        try:
            domain_ref = resource_api.get_domain_by_name(domain_name)
        except exception.DomainNotFound:
            LOG.warning('Invalid domain name (%s) found in config file name', domain_name)
            return
        domain_config = {}
        domain_config['cfg'] = cfg.ConfigOpts()
        keystone.conf.configure(conf=domain_config['cfg'])
        domain_config['cfg'](args=[], project='keystone', default_config_files=file_list, default_config_dirs=[])
        domain_config['driver'] = self._load_driver(domain_config)
        _assert_no_more_than_one_sql_driver(domain_config, file_list)
        self[domain_ref['id']] = domain_config

    def _setup_domain_drivers_from_files(self, standard_driver, resource_api):
        """Read the domain specific configuration files and load the drivers.

        Domain configuration files are stored in the domain config directory,
        and must be named of the form:

        keystone.<domain_name>.conf

        For each file, call the load config method where the domain_name
        will be turned into a domain_id and then:

        - Create a new config structure, adding in the specific additional
          options defined in this config file
        - Initialise a new instance of the required driver with this new config

        """
        conf_dir = CONF.identity.domain_config_dir
        if not os.path.exists(conf_dir):
            LOG.warning('Unable to locate domain config directory: %s', conf_dir)
            return
        for r, d, f in os.walk(conf_dir):
            for fname in f:
                if fname.startswith(DOMAIN_CONF_FHEAD) and fname.endswith(DOMAIN_CONF_FTAIL):
                    if fname.count('.') >= 2:
                        self._load_config_from_file(resource_api, [os.path.join(r, fname)], fname[len(DOMAIN_CONF_FHEAD):-len(DOMAIN_CONF_FTAIL)])
                    else:
                        LOG.debug('Ignoring file (%s) while scanning domain config directory', fname)

    def _load_config_from_database(self, domain_id, specific_config):

        def _assert_no_more_than_one_sql_driver(domain_id, new_config):
            """Ensure adding driver doesn't push us over the limit of 1.

            The checks we make in this method need to take into account that
            we may be in a multiple process configuration and ensure that
            any race conditions are avoided.

            """
            if not new_config['driver'].is_sql:
                PROVIDERS.domain_config_api.release_registration(domain_id)
                return
            domain_registered = 'Unknown'
            for attempt in range(REGISTRATION_ATTEMPTS):
                if PROVIDERS.domain_config_api.obtain_registration(domain_id, SQL_DRIVER):
                    LOG.debug('Domain %s successfully registered to use the SQL driver.', domain_id)
                    return
                try:
                    domain_registered = PROVIDERS.domain_config_api.read_registration(SQL_DRIVER)
                except exception.ConfigRegistrationNotFound:
                    msg = 'While attempting to register domain %(domain)s to use the SQL driver, another process released it, retrying (attempt %(attempt)s).'
                    LOG.debug(msg, {'domain': domain_id, 'attempt': attempt + 1})
                    continue
                if domain_registered == domain_id:
                    LOG.debug('While attempting to register domain %s to use the SQL driver, found that another process had already registered this domain. This is normal in multi-process configurations.', domain_id)
                    return
                try:
                    PROVIDERS.resource_api.get_domain(domain_registered)
                except exception.DomainNotFound:
                    msg = 'While attempting to register domain %(domain)s to use the SQL driver, found that it was already registered to a domain that no longer exists (%(old_domain)s). Removing this stale registration and retrying (attempt %(attempt)s).'
                    LOG.debug(msg, {'domain': domain_id, 'old_domain': domain_registered, 'attempt': attempt + 1})
                    PROVIDERS.domain_config_api.release_registration(domain_registered, type=SQL_DRIVER)
                    continue
                details = _('Config API entity at /domains/%s/config') % domain_id
                raise exception.MultipleSQLDriversInConfig(source=details)
            msg = _('Exceeded attempts to register domain %(domain)s to use the SQL driver, the last domain that appears to have had it is %(last_domain)s, giving up') % {'domain': domain_id, 'last_domain': domain_registered}
            raise exception.UnexpectedError(msg)
        domain_config = {}
        domain_config['cfg'] = cfg.ConfigOpts()
        keystone.conf.configure(conf=domain_config['cfg'])
        domain_config['cfg'](args=[], project='keystone', default_config_files=[], default_config_dirs=[])
        for group in specific_config:
            for option in specific_config[group]:
                domain_config['cfg'].set_override(option, specific_config[group][option], group)
        domain_config['cfg_overrides'] = specific_config
        domain_config['driver'] = self._load_driver(domain_config)
        _assert_no_more_than_one_sql_driver(domain_id, domain_config)
        self[domain_id] = domain_config

    def _setup_domain_drivers_from_database(self, standard_driver, resource_api):
        """Read domain specific configuration from database and load drivers.

        Domain configurations are stored in the domain-config backend,
        so we go through each domain to find those that have a specific config
        defined, and for those that do we:

        - Create a new config structure, overriding any specific options
          defined in the resource backend
        - Initialise a new instance of the required driver with this new config

        """
        for domain in resource_api.list_domains():
            domain_config_options = PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain['id'])
            if domain_config_options:
                self._load_config_from_database(domain['id'], domain_config_options)

    def setup_domain_drivers(self, standard_driver, resource_api):
        self.driver = standard_driver
        if CONF.identity.domain_configurations_from_database:
            self._setup_domain_drivers_from_database(standard_driver, resource_api)
        else:
            self._setup_domain_drivers_from_files(standard_driver, resource_api)
        self.configured = True

    def get_domain_driver(self, domain_id):
        self.check_config_and_reload_domain_driver_if_required(domain_id)
        if domain_id in self:
            return self[domain_id]['driver']

    def get_domain_conf(self, domain_id):
        self.check_config_and_reload_domain_driver_if_required(domain_id)
        if domain_id in self:
            return self[domain_id]['cfg']
        else:
            return CONF

    def reload_domain_driver(self, domain_id):
        if self.configured:
            if domain_id in self:
                self[domain_id]['driver'] = self._load_driver(self[domain_id])
            else:
                self.driver = self.driver()

    def check_config_and_reload_domain_driver_if_required(self, domain_id):
        """Check for, and load, any new domain specific config for this domain.

        This is only supported for the database-stored domain specific
        configuration.

        When the domain specific drivers were set up, we stored away the
        specific config for this domain that was available at that time. So we
        now read the current version and compare. While this might seem
        somewhat inefficient, the sensitive config call is cached, so should be
        light weight. More importantly, when the cache timeout is reached, we
        will get any config that has been updated from any other keystone
        process.

        This cache-timeout approach works for both multi-process and
        multi-threaded keystone configurations. In multi-threaded
        configurations, even though we might remove a driver object (that
        could be in use by another thread), this won't actually be thrown away
        until all references to it have been broken. When that other
        thread is released back and is restarted with another command to
        process, next time it accesses the driver it will pickup the new one.

        """
        if not CONF.identity.domain_specific_drivers_enabled or not CONF.identity.domain_configurations_from_database:
            return
        latest_domain_config = PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain_id)
        domain_config_in_use = domain_id in self
        if latest_domain_config:
            if not domain_config_in_use or latest_domain_config != self[domain_id]['cfg_overrides']:
                self._load_config_from_database(domain_id, latest_domain_config)
        elif domain_config_in_use:
            try:
                del self[domain_id]
            except KeyError:
                pass