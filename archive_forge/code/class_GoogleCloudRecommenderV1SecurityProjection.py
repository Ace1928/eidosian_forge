from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1SecurityProjection(_messages.Message):
    """Contains various ways of describing the impact on Security.

  Messages:
    DetailsValue: Additional security impact details that is provided by the
      recommender.

  Fields:
    details: Additional security impact details that is provided by the
      recommender.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DetailsValue(_messages.Message):
        """Additional security impact details that is provided by the
    recommender.

    Messages:
      AdditionalProperty: An additional property for a DetailsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    details = _messages.MessageField('DetailsValue', 1)