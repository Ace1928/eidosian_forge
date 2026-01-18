from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
def ReportPatchJobInstanceDetails(self, request, global_params=None):
    """Endpoint used by the agent to report back its state during a patch.
job. This endpoint will also return the patch job's state and
configurations that the agent needs to know in order to run or stop
patching.

This endpoint is only used by the agent. Using it in other ways may
affect the state of the active patch job and prevent the patches from
being correctly applied to this instance.

      Args:
        request: (OsconfigProjectsZonesInstancesReportPatchJobInstanceDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportPatchJobInstanceDetailsResponse) The response message.
      """
    config = self.GetMethodConfig('ReportPatchJobInstanceDetails')
    return self._RunMethod(config, request, global_params=global_params)