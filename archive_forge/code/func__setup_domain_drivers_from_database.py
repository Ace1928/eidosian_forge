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