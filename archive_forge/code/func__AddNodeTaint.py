from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddNodeTaint(vmware_node_config_group, for_update=False):
    """Adds a flag to specify the node taint in the node pool.

  Args:
    vmware_node_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    node_pool_create_help_text = 'Applies the given kubernetes taints on all nodes in the new node pool, which can\nbe used with tolerations for pod scheduling.\n\nTaint effect must be one of the following: `NoSchedule`, `PreferNoSchedule`, or `NoExecute`.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-taints=key1=val1:NoSchedule,key2=val2:PreferNoSchedule\n'
    node_pool_update_help_text = 'Replaces all the user specified Kubernetes taints on all nodes in an existing\nnode pool, which can be used with tolerations for pod scheduling.\n\nTaint effect must be one of the following: `NoSchedule`, `PreferNoSchedule`, or `NoExecute`.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-taints=key1=val1:NoSchedule,key2=val2:PreferNoSchedule\n'
    help_text = node_pool_update_help_text if for_update else node_pool_create_help_text
    vmware_node_config_group.add_argument('--node-taints', metavar='KEY=VALUE:EFFECT', help=help_text, type=arg_parsers.ArgDict())