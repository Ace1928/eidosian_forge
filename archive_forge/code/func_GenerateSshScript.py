from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1alpha2 import datamigration_v1alpha2_messages as messages
def GenerateSshScript(self, request, global_params=None):
    """Generate a SSH configuration script to configure the reverse SSH connectivity.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsGenerateSshScriptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SshScript) The response message.
      """
    config = self.GetMethodConfig('GenerateSshScript')
    return self._RunMethod(config, request, global_params=global_params)