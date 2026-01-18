from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OperationErrors(_messages.Message):
    """Database instance operation errors list wrapper.

  Fields:
    errors: The list of errors encountered while processing this operation.
    kind: This is always `sql#operationErrors`.
  """
    errors = _messages.MessageField('OperationError', 1, repeated=True)
    kind = _messages.StringField(2)