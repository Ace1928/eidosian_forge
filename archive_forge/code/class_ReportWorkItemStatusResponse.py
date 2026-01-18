from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportWorkItemStatusResponse(_messages.Message):
    """Response from a request to report the status of WorkItems.

  Messages:
    UnifiedWorkerResponseValue: Untranslated bag-of-bytes
      WorkProgressUpdateResponse for UnifiedWorker.

  Fields:
    unifiedWorkerResponse: Untranslated bag-of-bytes
      WorkProgressUpdateResponse for UnifiedWorker.
    workItemServiceStates: A set of messages indicating the service-side state
      for each WorkItem whose status was reported, in the same order as the
      WorkItemStatus messages in the ReportWorkItemStatusRequest which
      resulting in this response.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UnifiedWorkerResponseValue(_messages.Message):
        """Untranslated bag-of-bytes WorkProgressUpdateResponse for
    UnifiedWorker.

    Messages:
      AdditionalProperty: An additional property for a
        UnifiedWorkerResponseValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UnifiedWorkerResponseValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    unifiedWorkerResponse = _messages.MessageField('UnifiedWorkerResponseValue', 1)
    workItemServiceStates = _messages.MessageField('WorkItemServiceState', 2, repeated=True)