from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallPolicyAssociation(_messages.Message):
    """A FirewallPolicyAssociation object.

  Fields:
    attachmentTarget: The target that the firewall policy is attached to.
    displayName: [Output Only] Deprecated, please use short name instead. The
      display name of the firewall policy of the association.
    firewallPolicyId: [Output Only] The firewall policy ID of the association.
    name: The name for an association.
    shortName: [Output Only] The short name of the firewall policy of the
      association.
  """
    attachmentTarget = _messages.StringField(1)
    displayName = _messages.StringField(2)
    firewallPolicyId = _messages.StringField(3)
    name = _messages.StringField(4)
    shortName = _messages.StringField(5)