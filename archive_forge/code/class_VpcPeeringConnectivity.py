from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcPeeringConnectivity(_messages.Message):
    """The details of the VPC where the source database is located in Google
  Cloud. We will use this information to set up the VPC peering connection
  between Cloud SQL and this VPC.

  Fields:
    vpc: The name of the VPC network to peer with the Cloud SQL private
      network.
  """
    vpc = _messages.StringField(1)