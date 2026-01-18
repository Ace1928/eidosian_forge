from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
Generates an SBOM for the given resource.

      Args:
        request: (ContaineranalysisProjectsResourcesExportSBOMRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExportSBOMResponse) The response message.
      