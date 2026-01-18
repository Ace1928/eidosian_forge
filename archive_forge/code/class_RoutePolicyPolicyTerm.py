from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutePolicyPolicyTerm(_messages.Message):
    """A RoutePolicyPolicyTerm object.

  Fields:
    actions: CEL expressions to evaluate to modify a route when this term
      matches.
    match: CEL expression evaluated against a route to determine if this term
      applies. When not set, the term applies to all routes.
    priority: The evaluation priority for this term, which must be between 0
      (inclusive) and 2^31 (exclusive), and unique within the list.
  """
    actions = _messages.MessageField('Expr', 1, repeated=True)
    match = _messages.MessageField('Expr', 2)
    priority = _messages.IntegerField(3, variant=_messages.Variant.INT32)