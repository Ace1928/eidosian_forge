from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LabelClassWeightsValue(_messages.Message):
    """Weights associated with each label class, for rebalancing the training
    data. Only applicable for classification models.

    Messages:
      AdditionalProperty: An additional property for a LabelClassWeightsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        LabelClassWeightsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LabelClassWeightsValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
        key = _messages.StringField(1)
        value = _messages.FloatField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)