from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationTypeValueValuesEnum(_messages.Enum):
    """The type of index operation.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Unspecified. Never set by server.
      CREATING_INDEX: The operation is creating the index. Initiated by a
        `CreateIndex` call.
    """
    OPERATION_TYPE_UNSPECIFIED = 0
    CREATING_INDEX = 1