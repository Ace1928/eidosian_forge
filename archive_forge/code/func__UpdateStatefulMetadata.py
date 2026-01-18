from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_getter
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
@staticmethod
def _UpdateStatefulMetadata(messages, per_instance_config, update_stateful_metadata, remove_stateful_metadata):
    """Patch and return updated stateful metadata."""
    existing_metadata = []
    if per_instance_config.preservedState.metadata:
        existing_metadata = per_instance_config.preservedState.metadata.additionalProperties
    new_stateful_metadata = {metadata.key: metadata.value for metadata in existing_metadata}
    for metadata_key in remove_stateful_metadata or []:
        if metadata_key in new_stateful_metadata:
            del new_stateful_metadata[metadata_key]
        else:
            raise exceptions.InvalidArgumentException(parameter_name='--remove-stateful-metadata', message='stateful metadata key to remove `{0}` does not exist in the given instance config'.format(metadata_key))
    new_stateful_metadata.update(update_stateful_metadata)
    return new_stateful_metadata