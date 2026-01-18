from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetInstanceLabelsRequest(_messages.Message):
    """Request for setting instance labels.

  Messages:
    LabelsValue: Labels to apply to this instance. These can be later modified
      by the setLabels method

  Fields:
    labels: Labels to apply to this instance. These can be later modified by
      the setLabels method
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to apply to this instance. These can be later modified by the
    setLabels method

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