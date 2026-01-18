from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def UpdateMetadata(self, project, instance_ref, key, val):
    """Writes a key value pair to the metadata server.

    Args:
      project: The project string the instance is in.
      instance_ref: The instance the metadata server relates to.
      key: The string key to enter the data in.
      val: The string value to be written at the key.
    """
    messages = self._compute_client.MESSAGES_MODULE
    instance = self._compute_client.instances.Get(messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))
    existing_metadata = instance.metadata
    new_metadata = {key: val}
    self._compute_client.instances.SetMetadata(messages.ComputeInstancesSetMetadataRequest(instance=instance.name, metadata=metadata_utils.ConstructMetadataMessage(messages, metadata=new_metadata, existing_metadata=existing_metadata), project=project, zone=instance_ref.zone))