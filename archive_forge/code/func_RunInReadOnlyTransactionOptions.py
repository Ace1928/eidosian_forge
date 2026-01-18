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
def RunInReadOnlyTransactionOptions(options, function, *args, **kwargs):
    """Runs a function inside a read-only datastore transaction.

     A read-only transaction cannot perform writes, but may be able to execute
     more efficiently.

     Like RunInTransactionOptions, but with a read-only transaction.

  Args:
    options: TransactionOptions specifying options (number of retries, etc) for
      this transaction
    function: a function to be run inside the transaction on all remaining
      arguments
      *args: positional arguments for function.
      **kwargs: keyword arguments for function.

  Returns:
    the function's return value, if any

  Raises:
    TransactionFailedError, if the transaction could not be committed.
  """
    return _RunInTransactionInternal(options, datastore_rpc.TransactionMode.READ_ONLY, function, *args, **kwargs)