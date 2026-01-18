from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsGlobalHubsRejectSpokeRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsGlobalHubsRejectSpokeRequest
  object.

  Fields:
    googleCloudNetworkconnectivityV1betaRejectHubSpokeRequest: A
      GoogleCloudNetworkconnectivityV1betaRejectHubSpokeRequest resource to be
      passed as the request body.
    name: Required. The name of the hub from which to reject the spoke.
  """
    googleCloudNetworkconnectivityV1betaRejectHubSpokeRequest = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaRejectHubSpokeRequest', 1)
    name = _messages.StringField(2, required=True)