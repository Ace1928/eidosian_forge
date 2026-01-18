import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def acl_ref_relative(self):
    return ACL.get_acl_ref_from_entity_ref_relative(self.entity_uuid, self._parent_entity_path)