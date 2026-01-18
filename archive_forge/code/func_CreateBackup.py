from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def CreateBackup(self, request, global_params=None):
    """Creates a backup for a cloud sql instance. This API can only be used to create OnDemand backups.

      Args:
        request: (SqlBackupsCreateBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CreateBackup')
    return self._RunMethod(config, request, global_params=global_params)