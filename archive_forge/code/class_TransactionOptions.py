from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransactionOptions(_messages.Message):
    """Options for creating a new transaction.

  Fields:
    readOnly: The transaction can only be used for read operations.
    readWrite: The transaction can be used for both read and write operations.
  """
    readOnly = _messages.MessageField('ReadOnly', 1)
    readWrite = _messages.MessageField('ReadWrite', 2)