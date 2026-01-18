from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def _DatastoreExceptionFromErrorCodeAndDetail(error, detail):
    """Converts a datastore_pb.Error into a datastore_errors.Error.

  Args:
    error: A member of the datastore_pb.Error enumeration.
    detail: A string providing extra details about the error.

  Returns:
    An instance of a subclass of datastore_errors.Error.
  """
    exception_class = _DATASTORE_EXCEPTION_CLASSES.get(error, datastore_errors.Error)
    if detail is None:
        return exception_class()
    else:
        return exception_class(detail)