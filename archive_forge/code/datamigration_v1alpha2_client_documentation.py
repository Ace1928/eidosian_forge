from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1alpha2 import datamigration_v1alpha2_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (DatamigrationProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      