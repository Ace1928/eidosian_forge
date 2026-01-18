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
def _RunInTransactionInternal(options, mode, function, *args, **kwargs):
    """Runs a function inside a datastore transaction."""
    options = datastore_rpc.TransactionOptions(options)
    if IsInTransaction():
        if options.propagation in (None, datastore_rpc.TransactionOptions.NESTED):
            raise datastore_errors.BadRequestError('Nested transactions are not supported.')
        elif options.propagation is datastore_rpc.TransactionOptions.INDEPENDENT:
            txn_connection = _PopConnection()
            try:
                return _RunInTransactionInternal(options, mode, function, *args, **kwargs)
            finally:
                _PushConnection(txn_connection)
        return function(*args, **kwargs)
    if options.propagation is datastore_rpc.TransactionOptions.MANDATORY:
        raise datastore_errors.BadRequestError('Requires an existing transaction.')
    retries = options.retries
    if retries is None:
        retries = DEFAULT_TRANSACTION_RETRIES
    conn = _GetConnection()
    _PushConnection(None)
    previous_transaction = None
    transactional_conn = None
    try:
        for i in range(0, retries + 1):
            transactional_conn = conn.new_transaction(options, previous_transaction, mode)
            _SetConnection(transactional_conn)
            ok, result = _DoOneTry(function, args, kwargs)
            if ok:
                return result
            if i < retries:
                logging.warning('Transaction collision. Retrying... %s', '')
            if mode == datastore_rpc.TransactionMode.READ_WRITE:
                previous_transaction = transactional_conn.transaction
    finally:
        _PopConnection()
    if transactional_conn is not None:
        try:
            transactional_conn.rollback()
        except Exception:
            logging.exception('Exception sending Rollback:')
    raise datastore_errors.TransactionFailedError('The transaction could not be committed. Please try again.')