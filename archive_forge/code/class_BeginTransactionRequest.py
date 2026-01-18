from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeginTransactionRequest(_messages.Message):
    """The request for Firestore.BeginTransaction.

  Fields:
    options: The options for the transaction. Defaults to a read-write
      transaction.
  """
    options = _messages.MessageField('TransactionOptions', 1)