from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
Returns your permissions on a specified policy tag or taxonomy.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      