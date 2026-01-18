from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest
  object.

  Fields:
    googleCloudNetworkconnectivityV1betaAcceptHubSpokeRequest: A
      GoogleCloudNetworkconnectivityV1betaAcceptHubSpokeRequest resource to be
      passed as the request body.
    name: Required. The name of the hub into which to accept the spoke.
  """
    googleCloudNetworkconnectivityV1betaAcceptHubSpokeRequest = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaAcceptHubSpokeRequest', 1)
    name = _messages.StringField(2, required=True)