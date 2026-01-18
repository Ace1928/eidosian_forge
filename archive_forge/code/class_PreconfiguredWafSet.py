from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreconfiguredWafSet(_messages.Message):
    """A PreconfiguredWafSet object.

  Fields:
    expressionSets: List of entities that are currently supported for WAF
      rules.
  """
    expressionSets = _messages.MessageField('WafExpressionSet', 1, repeated=True)