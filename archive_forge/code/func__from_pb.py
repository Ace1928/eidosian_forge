from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
@classmethod
def _from_pb(cls, query_pb):
    kind = query_pb.has_kind() and query_pb.kind().decode('utf-8') or None
    ancestor = query_pb.has_ancestor() and query_pb.ancestor() or None
    filter_predicate = None
    if query_pb.filter_size() > 0:
        filter_predicate = CompositeFilter(CompositeFilter.AND, [PropertyFilter._from_pb(filter_pb) for filter_pb in query_pb.filter_list()])
    order = None
    if query_pb.order_size() > 0:
        order = CompositeOrder([PropertyOrder._from_pb(order_pb) for order_pb in query_pb.order_list()])
    group_by = None
    if query_pb.group_by_property_name_size() > 0:
        group_by = tuple((name.decode('utf-8') for name in query_pb.group_by_property_name_list()))
    read_time_us = None
    if query_pb.has_read_time_us():
        read_time_us = query_pb.read_time_us()
    return Query(app=query_pb.app().decode('utf-8'), namespace=query_pb.name_space().decode('utf-8'), kind=kind, ancestor=ancestor, filter_predicate=filter_predicate, order=order, group_by=group_by, read_time_us=read_time_us)