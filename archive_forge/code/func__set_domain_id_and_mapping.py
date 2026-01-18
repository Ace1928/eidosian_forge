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
def _set_domain_id_and_mapping(self, ref, domain_id, driver, entity_type):
    """Patch the domain_id/public_id into the resulting entity(ies).

        :param ref: the entity or list of entities to post process
        :param domain_id: the domain scope used for the call
        :param driver: the driver used to execute the call
        :param entity_type: whether this is a user or group

        :returns: post processed entity or list or entities

        Called to post-process the entity being returned, using a mapping
        to substitute a public facing ID as necessary. This method must
        take into account:

        - If the driver is not domain aware, then we must set the domain
          attribute of all entities irrespective of mapping.
        - If the driver does not support UUIDs, then we always want to provide
          a mapping, except for the special case of this being the default
          driver and backward_compatible_ids is set to True. This is to ensure
          that entity IDs do not change for an existing LDAP installation (only
          single domain/driver LDAP configurations were previously supported).
        - If the driver does support UUIDs, then we always create a mapping
          entry, but use the local UUID as the public ID.  The exception to
          this is that if we just have single driver (i.e. not using specific
          multi-domain configs), then we don't bother with the mapping at all.

        """
    conf = CONF.identity
    if not self._needs_post_processing(driver):
        return ref
    LOG.debug('ID Mapping - Domain ID: %(domain)s, Default Driver: %(driver)s, Domains: %(aware)s, UUIDs: %(generate)s, Compatible IDs: %(compat)s', {'domain': domain_id, 'driver': driver == self.driver, 'aware': driver.is_domain_aware(), 'generate': driver.generates_uuids(), 'compat': CONF.identity_mapping.backward_compatible_ids})
    if isinstance(ref, dict):
        return self._set_domain_id_and_mapping_for_single_ref(ref, domain_id, driver, entity_type, conf)
    elif isinstance(ref, list):
        return self._set_domain_id_and_mapping_for_list(ref, domain_id, driver, entity_type, conf)
    else:
        raise ValueError(_('Expected dict or list: %s') % type(ref))