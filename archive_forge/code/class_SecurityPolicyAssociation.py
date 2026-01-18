from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAssociation(_messages.Message):
    """A SecurityPolicyAssociation object.

  Fields:
    attachmentId: The resource that the security policy is attached to.
    displayName: [Output Only] The display name of the security policy of the
      association.
    name: The name for an association.
    securityPolicyId: [Output Only] The security policy ID of the association.
  """
    attachmentId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)
    securityPolicyId = _messages.StringField(4)