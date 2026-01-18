from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LabelSelector(_messages.Message):
    """A label selector is a label query over a set of resources. An empty
  label selector matches all objects.

  Messages:
    MatchLabelsValue: Optional. match_labels is a map of {key,value} pairs.
      Each {key,value} pair must match an existing label key and value exactly
      in order to satisfy the match.

  Fields:
    matchLabels: Optional. match_labels is a map of {key,value} pairs. Each
      {key,value} pair must match an existing label key and value exactly in
      order to satisfy the match.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MatchLabelsValue(_messages.Message):
        """Optional. match_labels is a map of {key,value} pairs. Each {key,value}
    pair must match an existing label key and value exactly in order to
    satisfy the match.

    Messages:
      AdditionalProperty: An additional property for a MatchLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MatchLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MatchLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    matchLabels = _messages.MessageField('MatchLabelsValue', 1)