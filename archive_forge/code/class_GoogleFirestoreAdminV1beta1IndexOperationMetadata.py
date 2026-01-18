from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1IndexOperationMetadata(_messages.Message):
    """Metadata for index operations. This metadata populates the metadata
  field of google.longrunning.Operation.

  Enums:
    OperationTypeValueValuesEnum: The type of index operation.

  Fields:
    cancelled: True if the [google.longrunning.Operation] was cancelled. If
      the cancellation is in progress, cancelled will be true but
      google.longrunning.Operation.done will be false.
    documentProgress: Progress of the existing operation, measured in number
      of documents.
    endTime: The time the operation ended, either successfully or otherwise.
      Unset if the operation is still active.
    index: The index resource that this operation is acting on. For example:
      `projects/{project_id}/databases/{database_id}/indexes/{index_id}`
    operationType: The type of index operation.
    startTime: The time that work began on the operation.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The type of index operation.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Unspecified. Never set by server.
      CREATING_INDEX: The operation is creating the index. Initiated by a
        `CreateIndex` call.
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        CREATING_INDEX = 1
    cancelled = _messages.BooleanField(1)
    documentProgress = _messages.MessageField('GoogleFirestoreAdminV1beta1Progress', 2)
    endTime = _messages.StringField(3)
    index = _messages.StringField(4)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 5)
    startTime = _messages.StringField(6)