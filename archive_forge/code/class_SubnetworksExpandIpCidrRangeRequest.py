from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubnetworksExpandIpCidrRangeRequest(_messages.Message):
    """A SubnetworksExpandIpCidrRangeRequest object.

  Fields:
    ipCidrRange: The IP (in CIDR format or netmask) of internal addresses that
      are legal on this Subnetwork. This range should be disjoint from other
      subnetworks within this network. This range can only be larger than
      (i.e. a superset of) the range previously defined before the update.
  """
    ipCidrRange = _messages.StringField(1)