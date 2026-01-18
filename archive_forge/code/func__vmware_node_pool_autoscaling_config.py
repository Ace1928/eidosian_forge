from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_node_pool_autoscaling_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareNodePoolAutoscalingConfig."""
    kwargs = {'minReplicas': flags.Get(args, 'min_replicas'), 'maxReplicas': flags.Get(args, 'max_replicas')}
    if any(kwargs.values()):
        return messages.VmwareNodePoolAutoscalingConfig(**kwargs)
    return None