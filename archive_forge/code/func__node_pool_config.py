from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import standalone_clusters
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages_module
def _node_pool_config(self, args: parser_extensions.Namespace):
    """Constructs proto message BareMetalStandaloneNodePoolConfig."""
    if 'node_configs_from_file' in args.GetSpecifiedArgsDict():
        node_configs = self._node_configs_from_file(args)
    else:
        node_configs = self._node_configs_from_flag(args)
    kwargs = {'nodeConfigs': node_configs, 'labels': self._node_labels(args), 'taints': self._node_taints(args), 'kubeletConfig': self._kubelet_config(args)}
    if any(kwargs.values()):
        return messages_module.BareMetalStandaloneNodePoolConfig(**kwargs)
    return None