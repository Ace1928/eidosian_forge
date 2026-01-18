from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __new_v3_property(self, v3_entity, is_indexed):
    if is_indexed:
        return v3_entity.add_property()
    else:
        return v3_entity.add_raw_property()