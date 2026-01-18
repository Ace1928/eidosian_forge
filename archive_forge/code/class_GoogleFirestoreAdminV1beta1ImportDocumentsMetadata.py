from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1ImportDocumentsMetadata(_messages.Message):
    """Metadata for ImportDocuments operations.

  Enums:
    OperationStateValueValuesEnum: The state of the import operation.

  Fields:
    collectionIds: Which collection ids are being imported.
    endTime: The time the operation ended, either successfully or otherwise.
      Unset if the operation is still active.
    inputUriPrefix: The location of the documents being imported.
    operationState: The state of the import operation.
    progressBytes: An estimate of the number of bytes processed.
    progressDocuments: An estimate of the number of documents processed.
    startTime: The time that work began on the operation.
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """The state of the import operation.

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
    collectionIds = _messages.StringField(1, repeated=True)
    endTime = _messages.StringField(2)
    inputUriPrefix = _messages.StringField(3)
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 4)
    progressBytes = _messages.MessageField('GoogleFirestoreAdminV1beta1Progress', 5)
    progressDocuments = _messages.MessageField('GoogleFirestoreAdminV1beta1Progress', 6)
    startTime = _messages.StringField(7)