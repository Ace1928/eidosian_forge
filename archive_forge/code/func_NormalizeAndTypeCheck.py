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
def NormalizeAndTypeCheck(arg, types):
    """Normalizes and type checks the given argument.

  Args:
    arg: an instance or iterable of the given type(s)
    types: allowed type or tuple of types

  Returns:
    A (list, bool) tuple. The list is a normalized, shallow copy of the
    argument. The boolean is True if the argument was a sequence, False
    if it was a single object.

  Raises:
    AssertionError: types includes list or tuple.
    BadArgumentError: arg is not an instance or sequence of one of the given
    types.
  """
    if not isinstance(types, (list, tuple)):
        types = (types,)
    assert list not in types and tuple not in types
    if isinstance(arg, types):
        return ([arg], False)
    else:
        if isinstance(arg, basestring):
            raise datastore_errors.BadArgumentError('Expected an instance or iterable of %s; received %s (a %s).' % (types, arg, typename(arg)))
        try:
            arg_list = list(arg)
        except TypeError:
            raise datastore_errors.BadArgumentError('Expected an instance or iterable of %s; received %s (a %s).' % (types, arg, typename(arg)))
        for val in arg_list:
            if not isinstance(val, types):
                raise datastore_errors.BadArgumentError('Expected one of %s; received %s (a %s).' % (types, val, typename(val)))
        return (arg_list, True)