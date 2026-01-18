from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2NetworkInterface(_messages.Message):
    """Direct VPC egress settings.

  Fields:
    network: Optional. The VPC network that the Cloud Run resource will be
      able to send traffic to. At least one of network or subnetwork must be
      specified. If both network and subnetwork are specified, the given VPC
      subnetwork must belong to the given VPC network. If network is not
      specified, it will be looked up from the subnetwork.
    subnetwork: Optional. The VPC subnetwork that the Cloud Run resource will
      get IPs from. At least one of network or subnetwork must be specified.
      If both network and subnetwork are specified, the given VPC subnetwork
      must belong to the given VPC network. If subnetwork is not specified,
      the subnetwork with the same name with the network will be used.
    tags: Optional. Network tags applied to this Cloud Run resource.
  """
    network = _messages.StringField(1)
    subnetwork = _messages.StringField(2)
    tags = _messages.StringField(3, repeated=True)