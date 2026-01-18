from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRuleOutput(_messages.Message):
    """A LearningGenaiRootRuleOutput object.

  Enums:
    DecisionValueValuesEnum:

  Fields:
    decision: A DecisionValueValuesEnum attribute.
    name: A string attribute.
  """

    class DecisionValueValuesEnum(_messages.Enum):
        """DecisionValueValuesEnum enum type.

    Values:
      NO_MATCH: This rule was not matched. When used in a ClassifierOutput,
        this means that no rules were matched.
      MATCH: This is a generic "match" message, indicating that a rule was
        triggered. Usually you would use this for a categorization classifier.
    """
        NO_MATCH = 0
        MATCH = 1
    decision = _messages.EnumField('DecisionValueValuesEnum', 1)
    name = _messages.StringField(2)