from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddControlPlaneNodeConfigs(bare_metal_node_config_group, is_update=False):
    """Adds flags to set the control plane node config.

  Args:
    bare_metal_node_config_group: The parent mutex group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    node_config_mutex_group = bare_metal_node_config_group.add_group(help='Populate control plane node config.', required=required, mutex=True)
    control_plane_node_configs_from_file_help_text = '\nPath of the YAML/JSON file that contains the control plane node configs.\n\nExamples:\n\n  nodeConfigs:\n  - nodeIP: 10.200.0.10\n    labels:\n      node1: label1\n      node2: label2\n  - nodeIP: 10.200.0.11\n    labels:\n      node3: label3\n      node4: label4\n\nList of supported fields in `nodeConfigs`\n\nKEY           | VALUE                     | NOTE\n--------------|---------------------------|---------------------------\nnodeIP        | string                    | required, mutable\nlabels        | one or more key-val pairs | optional, mutable\n\n'
    node_config_mutex_group.add_argument('--control-plane-node-configs-from-file', help=control_plane_node_configs_from_file_help_text, type=arg_parsers.YAMLFileContents(), hidden=True)
    node_config_mutex_group.add_argument('--control-plane-node-configs', help='Control plane node configuration.', action='append', type=arg_parsers.ArgDict(spec={'node-ip': str, 'labels': str}, required_keys=['node-ip']))