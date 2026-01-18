from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1 import datastream_v1_messages as messages
def Discover(self, request, global_params=None):
    """Use this method to discover a connection profile. The discover API call exposes the data objects and metadata belonging to the profile. Typically, a request returns children data objects of a parent data object that's optionally supplied in the request.

      Args:
        request: (DatastreamProjectsLocationsConnectionProfilesDiscoverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscoverConnectionProfileResponse) The response message.
      """
    config = self.GetMethodConfig('Discover')
    return self._RunMethod(config, request, global_params=global_params)