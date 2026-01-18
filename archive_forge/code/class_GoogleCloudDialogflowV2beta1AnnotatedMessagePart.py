from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1AnnotatedMessagePart(_messages.Message):
    """Represents a part of a message possibly annotated with an entity. The
  part can be an entity or purely a part of the message between two entities
  or message start/end.

  Fields:
    entityType: Optional. The [Dialogflow system entity
      type](https://cloud.google.com/dialogflow/docs/reference/system-
      entities) of this message part. If this is empty, Dialogflow could not
      annotate the phrase part with a system entity.
    formattedValue: Optional. The [Dialogflow system entity formatted value
      ](https://cloud.google.com/dialogflow/docs/reference/system-entities) of
      this message part. For example for a system entity of type `@sys.unit-
      currency`, this may contain: { "amount": 5, "currency": "USD" }
    text: Required. A part of a message possibly annotated with an entity.
  """
    entityType = _messages.StringField(1)
    formattedValue = _messages.MessageField('extra_types.JsonValue', 2)
    text = _messages.StringField(3)