import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def acl_ref(self):
    return ACL.get_acl_ref_from_entity_ref(self.entity_ref)