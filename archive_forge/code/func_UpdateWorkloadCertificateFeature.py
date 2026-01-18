from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workloadcertificate.v1alpha1 import workloadcertificate_v1alpha1_messages as messages
def UpdateWorkloadCertificateFeature(self, request, global_params=None):
    """Updates the `WorkloadCertificateFeature` resource of a given project.

      Args:
        request: (WorkloadcertificateProjectsLocationsGlobalUpdateWorkloadCertificateFeatureRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateWorkloadCertificateFeature')
    return self._RunMethod(config, request, global_params=global_params)