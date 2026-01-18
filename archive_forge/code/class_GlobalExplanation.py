from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlobalExplanation(_messages.Message):
    """Global explanations containing the top most important features after
  training.

  Fields:
    classLabel: Class label for this set of global explanations. Will be
      empty/null for binary logistic and linear regression models. Sorted
      alphabetically in descending order.
    explanations: A list of the top global explanations. Sorted by absolute
      value of attribution in descending order.
  """
    classLabel = _messages.StringField(1)
    explanations = _messages.MessageField('Explanation', 2, repeated=True)