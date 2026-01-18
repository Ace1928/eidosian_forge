from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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