from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgPolicyConstraintCustom(_messages.Message):
    """Message for Org Policy Custom Constraint.

  Fields:
    customConstraint: Required. Org Policy Custom Constraint.
    policyRules: Required. Org Policyspec rules.
  """
    customConstraint = _messages.MessageField('GoogleCloudSecuritypostureV1alphaCustomConstraint', 1)
    policyRules = _messages.MessageField('GoogleCloudSecuritypostureV1alphaPolicyRule', 2, repeated=True)