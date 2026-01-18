from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ActionMetricsValue(_messages.Message):
    """Action-based metrics. The map key is the action name which specified
    by the site owners at time of the "execute" client-side call.

    Messages:
      AdditionalProperty: An additional property for a ActionMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ActionMetricsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ActionMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudRecaptchaenterpriseV1ScoreDistribution attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1ScoreDistribution', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)