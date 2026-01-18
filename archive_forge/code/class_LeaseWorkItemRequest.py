from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LeaseWorkItemRequest(_messages.Message):
    """Request to lease WorkItems.

  Messages:
    UnifiedWorkerRequestValue: Untranslated bag-of-bytes WorkRequest from
      UnifiedWorker.

  Fields:
    currentWorkerTime: The current timestamp at the worker.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the WorkItem's job.
    requestedLeaseDuration: The initial lease period.
    unifiedWorkerRequest: Untranslated bag-of-bytes WorkRequest from
      UnifiedWorker.
    workItemTypes: Filter for WorkItem type.
    workerCapabilities: Worker capabilities. WorkItems might be limited to
      workers with specific capabilities.
    workerId: Identifies the worker leasing work -- typically the ID of the
      virtual machine running the worker.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UnifiedWorkerRequestValue(_messages.Message):
        """Untranslated bag-of-bytes WorkRequest from UnifiedWorker.

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
    requestedLeaseDuration = _messages.StringField(3)
    unifiedWorkerRequest = _messages.MessageField('UnifiedWorkerRequestValue', 4)
    workItemTypes = _messages.StringField(5, repeated=True)
    workerCapabilities = _messages.StringField(6, repeated=True)
    workerId = _messages.StringField(7)