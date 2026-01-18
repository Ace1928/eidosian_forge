from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterNatRuleAction(_messages.Message):
    """A RouterNatRuleAction object.

  Fields:
    sourceNatActiveIps: A list of URLs of the IP resources used for this NAT
      rule. These IP addresses must be valid static external IP addresses
      assigned to the project. This field is used for public NAT.
    sourceNatActiveRanges: A list of URLs of the subnetworks used as source
      ranges for this NAT Rule. These subnetworks must have purpose set to
      PRIVATE_NAT. This field is used for private NAT.
    sourceNatDrainIps: A list of URLs of the IP resources to be drained. These
      IPs must be valid static external IPs that have been assigned to the
      NAT. These IPs should be used for updating/patching a NAT rule only.
      This field is used for public NAT.
    sourceNatDrainRanges: A list of URLs of subnetworks representing source
      ranges to be drained. This is only supported on patch/update, and these
      subnetworks must have previously been used as active ranges in this NAT
      Rule. This field is used for private NAT.
  """
    sourceNatActiveIps = _messages.StringField(1, repeated=True)
    sourceNatActiveRanges = _messages.StringField(2, repeated=True)
    sourceNatDrainIps = _messages.StringField(3, repeated=True)
    sourceNatDrainRanges = _messages.StringField(4, repeated=True)