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
def PartitionString(value, separator):
    """Equivalent to python2.5 str.partition()
     TODO(user) use str.partition() when python 2.5 is adopted.

  Args:
    value: String to be partitioned
    separator: Separator string
  """
    index = value.find(separator)
    if index == -1:
        return (value, '', value[0:0])
    else:
        return (value[0:index], separator, value[index + len(separator):len(value)])