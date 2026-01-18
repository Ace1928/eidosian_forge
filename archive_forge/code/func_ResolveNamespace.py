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
def ResolveNamespace(namespace):
    """Validate app namespace, providing a default.

  If the argument is None, namespace_manager.get_namespace() is substituted.

  Args:
    namespace: The namespace argument value to be validated.

  Returns:
    The value of namespace, or the substituted default. The empty string is used
    to denote the empty namespace.

  Raises:
    BadArgumentError if the value is not a string.
  """
    if namespace is None:
        namespace = namespace_manager.get_namespace()
    else:
        namespace_manager.validate_namespace(namespace, datastore_errors.BadArgumentError)
    return namespace