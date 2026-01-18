from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def UpgradeAppliance(self, request, global_params=None):
    """Upgrades the appliance relate to this DatacenterConnector to the in-place updateable version.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDatacenterConnectorsUpgradeApplianceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpgradeAppliance')
    return self._RunMethod(config, request, global_params=global_params)