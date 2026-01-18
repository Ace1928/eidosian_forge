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
def _set_domain_id_and_mapping_for_list(self, ref_list, domain_id, driver, entity_type, conf):
    """Set domain id and mapping for a list of refs.

        The method modifies refs in-place.
        """
    if not ref_list:
        return []
    if not domain_id:
        domain_id = CONF.identity.default_domain_id
    if not driver.is_domain_aware():
        for ref in ref_list:
            ref['domain_id'] = domain_id
    if not self._is_mapping_needed(driver):
        return ref_list
    refs_map = {}
    for r in ref_list:
        refs_map[r['id'], entity_type, r['domain_id']] = r
    domain_mappings = PROVIDERS.id_mapping_api.get_domain_mapping_list(domain_id, entity_type=entity_type)
    for _mapping in domain_mappings:
        idx = (_mapping.local_id, _mapping.entity_type, _mapping.domain_id)
        try:
            ref = refs_map.pop(idx)
            ref['id'] = _mapping.public_id
        except KeyError:
            pass
    for ref in refs_map.values():
        local_entity = {'domain_id': ref['domain_id'], 'local_id': ref['id'], 'entity_type': entity_type}
        self._insert_new_public_id(local_entity, ref, driver)
    return ref_list