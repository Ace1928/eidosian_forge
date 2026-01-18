from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1IndexConfigDelta(_messages.Message):
    """Information about an index configuration change.

  Enums:
    ChangeTypeValueValuesEnum: Specifies how the index is changing.

  Fields:
    changeType: Specifies how the index is changing.
    index: The index being changed.
  """

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """Specifies how the index is changing.

    Values:
      CHANGE_TYPE_UNSPECIFIED: The type of change is not specified or known.
      ADD: The single field index is being added.
      REMOVE: The single field index is being removed.
    """
        CHANGE_TYPE_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 1)
    index = _messages.MessageField('GoogleFirestoreAdminV1Index', 2)