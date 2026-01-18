from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExtendedSourceContext(_messages.Message):
    """An ExtendedSourceContext is a SourceContext combined with additional
  details describing the context.

  Messages:
    LabelsValue: Labels with user defined metadata.

  Fields:
    context: Any source context.
    labels: Labels with user defined metadata.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels with user defined metadata.

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
    context = _messages.MessageField('SourceContext', 1)
    labels = _messages.MessageField('LabelsValue', 2)