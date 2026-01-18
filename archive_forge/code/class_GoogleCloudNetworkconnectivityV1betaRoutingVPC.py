from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaRoutingVPC(_messages.Message):
    """RoutingVPC contains information about the VPC networks associated with
  the spokes of a Network Connectivity Center hub.

  Fields:
    requiredForNewSiteToSiteDataTransferSpokes: Output only. If true,
      indicates that this VPC network is currently associated with spokes that
      use the data transfer feature (spokes where the
      site_to_site_data_transfer field is set to true). If you create new
      spokes that use data transfer, they must be associated with this VPC
      network. At most, one VPC network will have this field set to true.
    uri: The URI of the VPC network.
  """
    requiredForNewSiteToSiteDataTransferSpokes = _messages.BooleanField(1)
    uri = _messages.StringField(2)