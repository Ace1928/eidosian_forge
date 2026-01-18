from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LeaseWorkItemResponse(_messages.Message):
    """Response to a request to lease WorkItems.

  Messages:
    UnifiedWorkerResponseValue: Untranslated bag-of-bytes WorkResponse for
      UnifiedWorker.

  Fields:
    unifiedWorkerResponse: Untranslated bag-of-bytes WorkResponse for
      UnifiedWorker.
    workItems: A list of the leased WorkItems.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UnifiedWorkerResponseValue(_messages.Message):
        """Untranslated bag-of-bytes WorkResponse for UnifiedWorker.

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
    workItems = _messages.MessageField('WorkItem', 2, repeated=True)