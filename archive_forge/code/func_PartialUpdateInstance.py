from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
def PartialUpdateInstance(self, request, global_params=None):
    """Partially updates an instance within a project. This method can modify all fields of an Instance and is the preferred way to update an Instance.

      Args:
        request: (BigtableadminProjectsInstancesPartialUpdateInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PartialUpdateInstance')
    return self._RunMethod(config, request, global_params=global_params)