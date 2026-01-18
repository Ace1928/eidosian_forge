from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintTemplate(_messages.Message):
    """MembershipConstraintTemplate contains runtime status relevant to a
  single constraint template on a single member cluster.

  Fields:
    constraintTemplateRef: The constraint template this data refers to.
    description: annotations.description, may not be populated.
    membershipRef: The membership this data refers to.
    metadata: Membership-specific constraint template metadata.
    spec: Membership-specific constraint template spec.
    status: Membership-specific constraint template status.
  """
    constraintTemplateRef = _messages.MessageField('ConstraintTemplateRef', 1)
    description = _messages.StringField(2)
    membershipRef = _messages.MessageField('MembershipRef', 3)
    metadata = _messages.MessageField('MembershipConstraintTemplateMetadata', 4)
    spec = _messages.MessageField('MembershipConstraintTemplateSpec', 5)
    status = _messages.MessageField('MembershipConstraintTemplateStatus', 6)