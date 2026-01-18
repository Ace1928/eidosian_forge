from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetEffectiveOrgPolicyRequest(_messages.Message):
    """The request sent to the GetEffectiveOrgPolicy method.

  Fields:
    constraint: The name of the `Constraint` to compute the effective
      `Policy`.
  """
    constraint = _messages.StringField(1)