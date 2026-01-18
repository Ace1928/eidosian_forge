import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
@staticmethod
def FromPb(pb, validate_reserved_properties=True, default_kind='<not specified>'):
    """Static factory method. Returns the Entity representation of the
    given protocol buffer (datastore_pb.Entity).

    Args:
      pb: datastore_pb.Entity or str encoding of a datastore_pb.Entity
      validate_reserved_properties: deprecated
      default_kind: str, the kind to use if the pb has no key.

    Returns:
      Entity: the Entity representation of pb
    """
    if isinstance(pb, str):
        real_pb = entity_pb.EntityProto()
        real_pb.ParsePartialFromString(pb)
        pb = real_pb
    return Entity._FromPb(pb, require_valid_key=False, default_kind=default_kind)