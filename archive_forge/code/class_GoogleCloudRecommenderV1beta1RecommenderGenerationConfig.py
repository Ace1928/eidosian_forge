from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1RecommenderGenerationConfig(_messages.Message):
    """A Configuration to customize the generation of recommendations. Eg,
  customizing the lookback period considered when generating a recommendation.

  Messages:
    ParamsValue: Parameters for this RecommenderGenerationConfig. These
      configs can be used by or are applied to all subtypes.

  Fields:
    params: Parameters for this RecommenderGenerationConfig. These configs can
      be used by or are applied to all subtypes.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamsValue(_messages.Message):
        """Parameters for this RecommenderGenerationConfig. These configs can be
    used by or are applied to all subtypes.

    Messages:
      AdditionalProperty: An additional property for a ParamsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    params = _messages.MessageField('ParamsValue', 1)