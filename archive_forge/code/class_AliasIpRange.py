from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AliasIpRange(_messages.Message):
    """An alias IP range attached to an instance's network interface.

  Fields:
    ipCidrRange: Optional. The IP alias ranges to allocate for this interface.
    subnetworkRangeName: Optional. The name of a subnetwork secondary IP range
      from which to allocate an IP alias range. If not specified, the primary
      range of the subnetwork is used.
  """
    ipCidrRange = _messages.StringField(1)
    subnetworkRangeName = _messages.StringField(2)