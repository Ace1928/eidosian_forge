from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def UpdateBackup(self, request, global_params=None):
    """Updates the retention period and the description of the backup, currently restricted to final backups.

      Args:
        request: (SqlBackupsUpdateBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateBackup')
    return self._RunMethod(config, request, global_params=global_params)