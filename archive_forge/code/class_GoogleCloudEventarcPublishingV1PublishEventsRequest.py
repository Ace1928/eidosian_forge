from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudEventarcPublishingV1PublishEventsRequest(_messages.Message):
    """The request message for the PublishEvents method.

  Messages:
    EventsValueListEntry: A EventsValueListEntry object.

  Fields:
    events: The CloudEvents v1.0 events to publish. No other types are
      allowed. If this field is set, then the `text_events` fields must not be
      set.
    textEvents: The text representation of events to publish. CloudEvent v1.0
      in JSON format is the only allowed type. Refer to https://github.com/clo
      udevents/spec/blob/v1.0.2/cloudevents/formats/json-format.md for
      specification. If this field is set, then the `events` fields must not
      be set.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EventsValueListEntry(_messages.Message):
        """A EventsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a EventsValueListEntry
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EventsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    events = _messages.MessageField('EventsValueListEntry', 1, repeated=True)
    textEvents = _messages.StringField(2, repeated=True)