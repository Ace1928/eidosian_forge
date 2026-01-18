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
def RunInTransaction(function, *args, **kwargs):
    """Runs a function inside a datastore transaction.

     Runs the user-provided function inside transaction, retries default
     number of times.

    Args:
      function: a function to be run inside the transaction on all remaining
        arguments
      *args: positional arguments for function.
      **kwargs: keyword arguments for function.

  Returns:
    the function's return value, if any

  Raises:
    TransactionFailedError, if the transaction could not be committed.
  """
    return RunInTransactionOptions(None, function, *args, **kwargs)