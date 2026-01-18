from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v3_reference_has_id_or_name(self, v3_ref):
    """Determines if a v3 Reference specifies an ID or name.

    Args:
      v3_ref: an entity_pb.Reference

    Returns:
      boolean: True if the last path element specifies an ID or name.
    """
    path = v3_ref.path()
    assert path.element_size() >= 1
    last_element = path.element(path.element_size() - 1)
    return last_element.has_id() or last_element.has_name()