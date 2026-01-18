from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteBatchDmlRequest(_messages.Message):
    """The request for ExecuteBatchDml.

  Fields:
    requestOptions: Common options for this request.
    seqno: Required. A per-transaction sequence number used to identify this
      request. This field makes each request idempotent such that if the
      request is received multiple times, at most one will succeed. The
      sequence number must be monotonically increasing within the transaction.
      If a request arrives for the first time with an out-of-order sequence
      number, the transaction may be aborted. Replays of previously handled
      requests will yield the same response as the first execution.
    statements: Required. The list of statements to execute in this batch.
      Statements are executed serially, such that the effects of statement `i`
      are visible to statement `i+1`. Each statement must be a DML statement.
      Execution stops at the first failed statement; the remaining statements
      are not executed. Callers must provide at least one statement.
    transaction: Required. The transaction to use. Must be a read-write
      transaction. To protect against replays, single-use transactions are not
      supported. The caller must either supply an existing transaction ID or
      begin a new transaction.
  """
    requestOptions = _messages.MessageField('RequestOptions', 1)
    seqno = _messages.IntegerField(2)
    statements = _messages.MessageField('Statement', 3, repeated=True)
    transaction = _messages.MessageField('TransactionSelector', 4)