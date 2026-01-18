from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1beta1CommonMetadata(_messages.Message):
    """Metadata common to all Datastore Admin operations.

  Enums:
    OperationTypeValueValuesEnum: The type of the operation. Can be used as a
      filter in ListOperationsRequest.
    StateValueValuesEnum: The current state of the Operation.

  Messages:
    LabelsValue: The client-assigned labels which were provided when the
      operation was created. May also include additional labels.

  Fields:
    endTime: The time the operation ended, either successfully or otherwise.
    labels: The client-assigned labels which were provided when the operation
      was created. May also include additional labels.
    operationType: The type of the operation. Can be used as a filter in
      ListOperationsRequest.
    startTime: The time that work began on the operation.
    state: The current state of the Operation.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The type of the operation. Can be used as a filter in
    ListOperationsRequest.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Unspecified.
      EXPORT_ENTITIES: ExportEntities.
      IMPORT_ENTITIES: ImportEntities.
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        EXPORT_ENTITIES = 1
        IMPORT_ENTITIES = 2

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the Operation.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      INITIALIZING: Request is being prepared for processing.
      PROCESSING: Request is actively being processed.
      CANCELLING: Request is in the process of being cancelled after user
        called google.longrunning.Operations.CancelOperation on the operation.
      FINALIZING: Request has been processed and is in its finalization stage.
      SUCCESSFUL: Request has completed successfully.
      FAILED: Request has finished being processed, but encountered an error.
      CANCELLED: Request has finished being cancelled after user called
        google.longrunning.Operations.CancelOperation.
    """
        STATE_UNSPECIFIED = 0
        INITIALIZING = 1
        PROCESSING = 2
        CANCELLING = 3
        FINALIZING = 4
        SUCCESSFUL = 5
        FAILED = 6
        CANCELLED = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The client-assigned labels which were provided when the operation was
    created. May also include additional labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    endTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 3)
    startTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)