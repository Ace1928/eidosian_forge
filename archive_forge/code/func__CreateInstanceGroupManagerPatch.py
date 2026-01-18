from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils as mig_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_instance_groups_flags
def _CreateInstanceGroupManagerPatch(self, args, client):
    """Creates IGM resource patch."""
    mig_utils.RegisterCustomInstancePropertiesPatchEncoders(client)
    metadata = args.metadata or []
    labels = args.labels or []
    return client.messages.InstanceGroupManager(allInstancesConfig=client.messages.InstanceGroupManagerAllInstancesConfig(properties=client.messages.InstancePropertiesPatch(metadata=client.messages.InstancePropertiesPatch.MetadataValue(additionalProperties=[client.messages.InstancePropertiesPatch.MetadataValue.AdditionalProperty(key=key, value=None) for key in metadata]), labels=client.messages.InstancePropertiesPatch.LabelsValue(additionalProperties=[client.messages.InstancePropertiesPatch.LabelsValue.AdditionalProperty(key=key, value=None) for key in labels]))))