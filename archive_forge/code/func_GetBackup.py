from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def GetBackup(self, request, global_params=None):
    """Retrieves a resource containing information about a backup.

      Args:
        request: (SqlBackupsGetBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
    config = self.GetMethodConfig('GetBackup')
    return self._RunMethod(config, request, global_params=global_params)