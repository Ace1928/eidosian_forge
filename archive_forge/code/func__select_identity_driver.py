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
def _select_identity_driver(self, domain_id):
    """Choose a backend driver for the given domain_id.

        :param domain_id: The domain_id for which we want to find a driver.  If
                          the domain_id is specified as None, then this means
                          we need a driver that handles multiple domains.

        :returns: chosen backend driver

        If there is a specific driver defined for this domain then choose it.
        If the domain is None, or there no specific backend for the given
        domain is found, then we chose the default driver.

        """
    if domain_id is None:
        driver = self.driver
    else:
        driver = self.domain_configs.get_domain_driver(domain_id) or self.driver
    if not driver.is_domain_aware() and driver == self.driver and (domain_id != CONF.identity.default_domain_id) and (domain_id is not None):
        LOG.warning('Found multiple domains being mapped to a driver that does not support that (e.g. LDAP) - Domain ID: %(domain)s, Default Driver: %(driver)s', {'domain': domain_id, 'driver': driver == self.driver})
        raise exception.DomainNotFound(domain_id=domain_id)
    return driver