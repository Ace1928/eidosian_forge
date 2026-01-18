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
def ValidatePropertyKey(name, value):
    """Raises an exception if the supplied datastore.Key instance is invalid.

  Args:
    name: Name of the property this is for.
    value: A datastore.Key instance.

  Raises:
    datastore_errors.BadValueError if the value is invalid.
  """
    if not value.has_id_or_name():
        raise datastore_errors.BadValueError('Incomplete key found for reference property %s.' % name)