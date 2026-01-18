from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LabelSelectorRequirement(_messages.Message):
    """A label selector requirement is a selector that contains values, a key,
  and an operator that relates the key and values.

  Fields:
    key: key is the label key that the selector applies to. +patchMergeKey=key
      +patchStrategy=merge
    operator: operator represents a key's relationship to a set of values.
      Valid operators are In, NotIn, Exists and DoesNotExist.
    values: values is an array of string values. If the operator is In or
      NotIn, the values array must be non-empty. If the operator is Exists or
      DoesNotExist, the values array must be empty. This array is replaced
      during a strategic merge patch. +optional
  """
    key = _messages.StringField(1)
    operator = _messages.StringField(2)
    values = _messages.StringField(3)