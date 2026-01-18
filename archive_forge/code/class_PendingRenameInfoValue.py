from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PendingRenameInfoValue(_messages.Message):
    """Only present if the folder is part of an ongoing rename folder
    operation. Contains information which can be used to query the operation
    status.

    Fields:
      operationId: The ID of the rename folder operation.
    """
    operationId = _messages.StringField(1)