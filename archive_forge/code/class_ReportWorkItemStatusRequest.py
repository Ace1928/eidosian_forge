from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportWorkItemStatusRequest(_messages.Message):
    """Request to report the status of WorkItems.

  Messages:
    UnifiedWorkerRequestValue: Untranslated bag-of-bytes
      WorkProgressUpdateRequest from UnifiedWorker.

  Fields:
    currentWorkerTime: The current timestamp at the worker.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the WorkItem's job.
    unifiedWorkerRequest: Untranslated bag-of-bytes WorkProgressUpdateRequest
      from UnifiedWorker.
    workItemStatuses: The order is unimportant, except that the order of the
      WorkItemServiceState messages in the ReportWorkItemStatusResponse
      corresponds to the order of WorkItemStatus messages here.
    workerId: The ID of the worker reporting the WorkItem status. If this does
      not match the ID of the worker which the Dataflow service believes
      currently has the lease on the WorkItem, the report will be dropped
      (with an error response).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UnifiedWorkerRequestValue(_messages.Message):
        """Untranslated bag-of-bytes WorkProgressUpdateRequest from
    UnifiedWorker.

    Messages:
      AdditionalProperty: An additional property for a
        UnifiedWorkerRequestValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UnifiedWorkerRequestValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    currentWorkerTime = _messages.StringField(1)
    location = _messages.StringField(2)
    unifiedWorkerRequest = _messages.MessageField('UnifiedWorkerRequestValue', 3)
    workItemStatuses = _messages.MessageField('WorkItemStatus', 4, repeated=True)
    workerId = _messages.StringField(5)