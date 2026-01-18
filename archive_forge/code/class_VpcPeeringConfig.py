from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcPeeringConfig(_messages.Message):
    """The VPC peering configuration is used to create VPC peering with the
  consumer's VPC.

  Fields:
    subnet: Required. A free subnet for peering. (CIDR of /29)
    vpcName: Required. Fully qualified name of the VPC that Database Migration
      Service will peer to.
  """
    subnet = _messages.StringField(1)
    vpcName = _messages.StringField(2)