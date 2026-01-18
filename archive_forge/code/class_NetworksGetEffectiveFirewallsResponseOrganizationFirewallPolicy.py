from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksGetEffectiveFirewallsResponseOrganizationFirewallPolicy(_messages.Message):
    """A pruned SecurityPolicy containing ID and any applicable firewall rules.

  Fields:
    id: [Output Only] The unique identifier for the security policy. This
      identifier is defined by the server.
    rules: The rules that apply to the network.
  """
    id = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    rules = _messages.MessageField('SecurityPolicyRule', 2, repeated=True)