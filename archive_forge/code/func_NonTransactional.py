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
@datastore_rpc._positional(1)
def NonTransactional(_func=None, allow_existing=True):
    """A decorator that insures a function is run outside a transaction.

  If there is an existing transaction (and allow_existing=True), the existing
  transaction is paused while the function is executed.

  Args:
    _func: do not use
    allow_existing: If false, throw an exception if called from within a
      transaction

  Returns:
    A wrapper for the decorated function that ensures it runs outside a
    transaction.
  """
    if _func is not None:
        return NonTransactional()(_func)

    def outer_wrapper(func):

        def inner_wrapper(*args, **kwds):
            if not IsInTransaction():
                return func(*args, **kwds)
            if not allow_existing:
                raise datastore_errors.BadRequestError('Function cannot be called from within a transaction.')
            txn_connection = _PopConnection()
            try:
                return func(*args, **kwds)
            finally:
                _PushConnection(txn_connection)
        return inner_wrapper
    return outer_wrapper