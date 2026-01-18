from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.file.v1p1alpha1 import file_v1p1alpha1_messages as messages
def CopyInstance(self, request, global_params=None):
    """Copies the fileshare content of a Basic instance to a High Scale or Enterprise tier instance. If the source instance is being written to during the copy, the copy will not be a consistent snapshot of the fileshare. If the target instance already has files, these files will be overwritten if the source instance has the same file but with different checksum values. Files that exist in the target but not in the source will be deleted. Hard links are copied as separate files. POSIX ACLs are not copied. The source and target instances must be on the same VPC and using the same `connect_mode`.

      Args:
        request: (FileProjectsLocationsInstancesCopyInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CopyInstance')
    return self._RunMethod(config, request, global_params=global_params)