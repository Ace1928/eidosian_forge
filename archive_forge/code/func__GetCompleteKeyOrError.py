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
def _GetCompleteKeyOrError(arg):
    """Expects an Entity or a Key, and returns the corresponding Key. Raises
  BadArgumentError or BadKeyError if arg is a different type or is incomplete.

  Args:
    arg: Entity or Key

  Returns:
    Key
  """
    if isinstance(arg, Key):
        key = arg
    elif isinstance(arg, basestring):
        key = Key(arg)
    elif isinstance(arg, Entity):
        key = arg.key()
    elif not isinstance(arg, Key):
        raise datastore_errors.BadArgumentError('Expects argument to be an Entity or Key; received %s (a %s).' % (arg, typename(arg)))
    assert isinstance(key, Key)
    if not key.has_id_or_name():
        raise datastore_errors.BadKeyError('Key %r is not complete.' % key)
    return key