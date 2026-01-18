from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelMetadata(_messages.Message):
    """The metadata about the models in a given region for a specific locale.
  Currently this is just the features of the model

  Messages:
    ModelFeaturesValue: Map of the model name -> features of that model

  Fields:
    modelFeatures: Map of the model name -> features of that model
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModelFeaturesValue(_messages.Message):
        """Map of the model name -> features of that model

    Messages:
      AdditionalProperty: An additional property for a ModelFeaturesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ModelFeaturesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ModelFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A ModelFeatures attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ModelFeatures', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    modelFeatures = _messages.MessageField('ModelFeaturesValue', 1)