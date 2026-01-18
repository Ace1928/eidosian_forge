from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MatchTargetLabelsValue(_messages.Message):
    """Optional. Deploy parameters are applied to targets with match labels.
    If unspecified, deploy parameters are applied to all targets (including
    child targets of a multi-target).

    Messages:
      AdditionalProperty: An additional property for a MatchTargetLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MatchTargetLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MatchTargetLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)