from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateTypeValueValuesEnum(_messages.Enum):
    """Specify the type of the config change.

    Values:
      UPDATE_TYPE_UNSPECIFIED: Default update type.
      INSERT: Indicates the update type is insertion.
      DELETE: Indicates the update type is deletion.
      UPDATE: Indicates the update type is modification.
    """
    UPDATE_TYPE_UNSPECIFIED = 0
    INSERT = 1
    DELETE = 2
    UPDATE = 3