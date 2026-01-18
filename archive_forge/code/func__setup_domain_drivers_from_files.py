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