from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class StateError(_messages.Message):
    """Describes an error related to the current state of the Execution
  resource.

  Enums:
    TypeValueValuesEnum: The type of this state error.

  Fields:
    details: Provides specifics about the error.
    type: The type of this state error.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of this state error.

    Values:
      TYPE_UNSPECIFIED: No type specified.
      KMS_ERROR: Caused by an issue with KMS.
    """
        TYPE_UNSPECIFIED = 0
        KMS_ERROR = 1
    details = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)