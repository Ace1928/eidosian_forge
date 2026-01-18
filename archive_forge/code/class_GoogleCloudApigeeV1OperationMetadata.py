from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OperationMetadata(_messages.Message):
    """Metadata describing an Operation.

  Enums:
    OperationTypeValueValuesEnum:
    StateValueValuesEnum:

  Fields:
    operationType: A OperationTypeValueValuesEnum attribute.
    progress: Progress of the operation.
    state: A StateValueValuesEnum attribute.
    targetResourceName: Name of the resource for which the operation is
      operating on.
    warnings: Warnings encountered while executing the operation.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """OperationTypeValueValuesEnum enum type.

    Values:
      OPERATION_TYPE_UNSPECIFIED: <no description>
      INSERT: <no description>
      DELETE: <no description>
      UPDATE: <no description>
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        INSERT = 1
        DELETE = 2
        UPDATE = 3

    class StateValueValuesEnum(_messages.Enum):
        """StateValueValuesEnum enum type.

    Values:
      STATE_UNSPECIFIED: <no description>
      NOT_STARTED: <no description>
      IN_PROGRESS: <no description>
      FINISHED: <no description>
    """
        STATE_UNSPECIFIED = 0
        NOT_STARTED = 1
        IN_PROGRESS = 2
        FINISHED = 3
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 1)
    progress = _messages.MessageField('GoogleCloudApigeeV1OperationMetadataProgress', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    targetResourceName = _messages.StringField(4)
    warnings = _messages.StringField(5, repeated=True)