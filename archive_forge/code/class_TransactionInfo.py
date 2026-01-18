from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransactionInfo(_messages.Message):
    """[Alpha] Information of a multi-statement transaction.

  Fields:
    transactionId: Output only. [Alpha] Id of the transaction.
  """
    transactionId = _messages.StringField(1)