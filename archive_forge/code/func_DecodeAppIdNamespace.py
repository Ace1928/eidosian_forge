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
def DecodeAppIdNamespace(app_namespace_str):
    """Decodes app_namespace_str into an (app_id, namespace) pair.

  This method is the reverse of EncodeAppIdNamespace and is needed for
  datastore_file_stub.

  Args:
    app_namespace_str: An encoded app_id, namespace pair created by
      EncodeAppIdNamespace

  Returns:
    (app_id, namespace) pair encoded in app_namespace_str
  """
    sep = app_namespace_str.find(_NAMESPACE_SEPARATOR)
    if sep < 0:
        return (app_namespace_str, '')
    else:
        return (app_namespace_str[0:sep], app_namespace_str[sep + 1:])