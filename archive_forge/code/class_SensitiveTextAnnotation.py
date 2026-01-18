from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SensitiveTextAnnotation(_messages.Message):
    """A TextAnnotation specifies a text range that includes sensitive
  information.

  Messages:
    DetailsValue: Maps from a resource slice. For example, FHIR resource field
      path to a set of sensitive text findings. For example,
      Appointment.Narrative text1 --> {findings_1, findings_2, findings_3}

  Fields:
    details: Maps from a resource slice. For example, FHIR resource field path
      to a set of sensitive text findings. For example, Appointment.Narrative
      text1 --> {findings_1, findings_2, findings_3}
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DetailsValue(_messages.Message):
        """Maps from a resource slice. For example, FHIR resource field path to a
    set of sensitive text findings. For example, Appointment.Narrative text1
    --> {findings_1, findings_2, findings_3}

    Messages:
      AdditionalProperty: An additional property for a DetailsValue object.

    Fields:
      additionalProperties: Additional properties of type DetailsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A Detail attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Detail', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    details = _messages.MessageField('DetailsValue', 1)