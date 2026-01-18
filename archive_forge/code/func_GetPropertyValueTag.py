from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetPropertyValueTag(value_pb):
    """Returns the tag constant associated with the given entity_pb.PropertyValue.
  """
    if value_pb.has_booleanvalue():
        return entity_pb.PropertyValue.kbooleanValue
    elif value_pb.has_doublevalue():
        return entity_pb.PropertyValue.kdoubleValue
    elif value_pb.has_int64value():
        return entity_pb.PropertyValue.kint64Value
    elif value_pb.has_pointvalue():
        return entity_pb.PropertyValue.kPointValueGroup
    elif value_pb.has_referencevalue():
        return entity_pb.PropertyValue.kReferenceValueGroup
    elif value_pb.has_stringvalue():
        return entity_pb.PropertyValue.kstringValue
    elif value_pb.has_uservalue():
        return entity_pb.PropertyValue.kUserValueGroup
    else:
        return 0