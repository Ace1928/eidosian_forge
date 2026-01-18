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
def _node_configs_from_flag(self, args: parser_extensions.Namespace):
    """Constructs proto message field node_configs."""
    node_configs = []
    node_config_flag_value = getattr(args, 'node_configs', None)
    if node_config_flag_value:
        for node_config in node_config_flag_value:
            node_configs.append(self.standalone_node_config(node_config))
    return node_configs