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
def _vmware_node_pool(self, args: parser_extensions.Namespace) -> messages.VmwareNodePool:
    """Constructs proto message VmwareNodePool."""
    kwargs = {'name': self._node_pool_name(args), 'displayName': flags.Get(args, 'display_name'), 'annotations': self._annotations(args), 'config': self._vmware_node_config(args), 'onPremVersion': flags.Get(args, 'version'), 'nodePoolAutoscaling': self._vmware_node_pool_autoscaling_config(args)}
    return messages.VmwareNodePool(**kwargs)