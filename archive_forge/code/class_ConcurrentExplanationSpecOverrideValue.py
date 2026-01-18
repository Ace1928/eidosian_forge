from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConcurrentExplanationSpecOverrideValue(_messages.Message):
    """Optional. This field is the same as the one above, but supports
    multiple explanations to occur in parallel. The key can be any string.
    Each override will be run against the model, then its explanations will be
    grouped together. Note - these explanations are run **In Addition** to the
    default Explanation in the deployed model.

    Messages:
      AdditionalProperty: An additional property for a
        ConcurrentExplanationSpecOverrideValue object.

    Fields:
      additionalProperties: Additional properties of type
        ConcurrentExplanationSpecOverrideValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ConcurrentExplanationSpecOverrideValue
      object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1ExplanationSpecOverride
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationSpecOverride', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)