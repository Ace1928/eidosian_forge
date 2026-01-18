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
def RunInTransactionOptions(options, function, *args, **kwargs):
    """Runs a function inside a datastore transaction.

  Runs the user-provided function inside a full-featured, ACID datastore
  transaction. Every Put, Get, and Delete call in the function is made within
  the transaction. All entities involved in these calls must belong to the
  same entity group. Queries are supported as long as they specify an
  ancestor belonging to the same entity group.

  The trailing arguments are passed to the function as positional arguments.
  If the function returns a value, that value will be returned by
  RunInTransaction. Otherwise, it will return None.

  The function may raise any exception to roll back the transaction instead of
  committing it. If this happens, the transaction will be rolled back and the
  exception will be re-raised up to RunInTransaction's caller.

  If you want to roll back intentionally, but don't have an appropriate
  exception to raise, you can raise an instance of datastore_errors.Rollback.
  It will cause a rollback, but will *not* be re-raised up to the caller.

  The function may be run more than once, so it should be idempotent. It
  should avoid side effects, and it shouldn't have *any* side effects that
  aren't safe to occur multiple times. This includes modifying the arguments,
  since they persist across invocations of the function. However, this doesn't
  include Put, Get, and Delete calls, of course.

  Example usage:

  > def decrement(key, amount=1):
  >   counter = datastore.Get(key)
  >   counter['count'] -= amount
  >   if counter['count'] < 0:    # don't let the counter go negative
  >     raise datastore_errors.Rollback()
  >   datastore.Put(counter)
  >
  > counter = datastore.Query('Counter', {'name': 'foo'})
  > datastore.RunInTransaction(decrement, counter.key(), amount=5)

  Transactions satisfy the traditional ACID properties. They are:

  - Atomic. All of a transaction's operations are executed or none of them are.

  - Consistent. The datastore's state is consistent before and after a
  transaction, whether it committed or rolled back. Invariants such as
  "every entity has a primary key" are preserved.

  - Isolated. Transactions operate on a snapshot of the datastore. Other
  datastore operations do not see intermediated effects of the transaction;
  they only see its effects after it has committed.

  - Durable. On commit, all writes are persisted to the datastore.

  Nested transactions are not supported.

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
    return _RunInTransactionInternal(options, datastore_rpc.TransactionMode.READ_WRITE, function, *args, **kwargs)