from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPeeringsGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsNetworkPeeringsGetRequest object.

  Fields:
    name: Required. The resource name of the network peering to retrieve.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/global/networkPeerings/my-peering`
  """
    name = _messages.StringField(1, required=True)