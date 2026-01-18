from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Hl7V2Store(_messages.Message):
    """Represents an HL7v2 store.

  Messages:
    LabelsValue: User-supplied key-value pairs used to organize HL7v2 stores.
      Label keys must be between 1 and 63 characters long, have a UTF-8
      encoding of maximum 128 bytes, and must conform to the following PCRE
      regular expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must
      be between 1 and 63 characters long, have a UTF-8 encoding of maximum
      128 bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.

  Fields:
    labels: User-supplied key-value pairs used to organize HL7v2 stores. Label
      keys must be between 1 and 63 characters long, have a UTF-8 encoding of
      maximum 128 bytes, and must conform to the following PCRE regular
      expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be
      between 1 and 63 characters long, have a UTF-8 encoding of maximum 128
      bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.
    name: Identifier. Resource name of the HL7v2 store, of the form `projects/
      {project_id}/locations/{location_id}/datasets/{dataset_id}/hl7V2Stores/{
      hl7v2_store_id}`.
    notificationConfig: The notification destination all messages (both Ingest
      & Create) are published on. Only the message name is sent as part of the
      notification. If this is unset, no notifications are sent. Supplied by
      the client.
    parserConfig: The configuration for the parser. It determines how the
      server parses the messages.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-supplied key-value pairs used to organize HL7v2 stores. Label
    keys must be between 1 and 63 characters long, have a UTF-8 encoding of
    maximum 128 bytes, and must conform to the following PCRE regular
    expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be between
    1 and 63 characters long, have a UTF-8 encoding of maximum 128 bytes, and
    must conform to the following PCRE regular expression:
    [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated with
    a given store.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    name = _messages.StringField(2)
    notificationConfig = _messages.MessageField('NotificationConfig', 3)
    parserConfig = _messages.MessageField('ParserConfig', 4)