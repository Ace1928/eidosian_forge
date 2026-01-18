from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistributionExemplar(_messages.Message):
    """Exemplars are example points that may be used to annotate aggregated
  distribution values. They are metadata that gives information about a
  particular value added to a Distribution bucket, such as a trace ID that was
  active when a value was added. They may contain further information, such as
  a example values and timestamps, origin, etc.

  Messages:
    AttachmentsValueListEntry: A AttachmentsValueListEntry object.

  Fields:
    attachments: Contextual information about the example value. Examples are:
      Trace: type.googleapis.com/google.monitoring.v3.SpanContext Literal
      string: type.googleapis.com/google.protobuf.StringValue Labels dropped
      during aggregation:
      type.googleapis.com/google.monitoring.v3.DroppedLabels There may be only
      a single attachment of any given message type in a single exemplar, and
      this is enforced by the system.
    timestamp: The observation (sampling) time of the above value.
    value: Value of the exemplar point. This value determines to which bucket
      the exemplar belongs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttachmentsValueListEntry(_messages.Message):
        """A AttachmentsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        AttachmentsValueListEntry object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttachmentsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attachments = _messages.MessageField('AttachmentsValueListEntry', 1, repeated=True)
    timestamp = _messages.StringField(2)
    value = _messages.FloatField(3)