from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
Converts a v3 Query order to a v4 PropertyOrder.

    Args:
      v3_order: a datastore_pb.Query.Order
      v4_order: a datastore_v4_pb.PropertyOrder to populate
    