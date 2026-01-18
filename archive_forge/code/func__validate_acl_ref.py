import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def _validate_acl_ref(self, entity_ref):
    if entity_ref is None:
        raise ValueError('Expected secret or container URI is not specified.')
    entity_ref = entity_ref.rstrip('/')
    entity_type = ACL.identify_ref_type(entity_ref)
    entity_class = ACLManager.acl_class_map.get(entity_type)
    acl_entity = entity_class(api=self._api, entity_ref=entity_ref)
    acl_entity.validate_input_ref()
    return acl_entity