from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WafExpressionSet(_messages.Message):
    """A WafExpressionSet object.

  Fields:
    aliases: A list of alternate IDs. The format should be: - E.g. XSS-stable
      Generic suffix like "stable" is particularly useful if a policy likes to
      avail newer set of expressions without having to change the policy. A
      given alias name can't be used for more than one entity set.
    expressions: List of available expressions.
    id: Google specified expression set ID. The format should be: - E.g.
      XSS-20170329 required
  """
    aliases = _messages.StringField(1, repeated=True)
    expressions = _messages.MessageField('WafExpressionSetExpression', 2, repeated=True)
    id = _messages.StringField(3)