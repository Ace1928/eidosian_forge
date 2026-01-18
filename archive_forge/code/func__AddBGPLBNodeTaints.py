from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBNodeTaints(bare_metal_bgp_lb_node_config):
    """Adds a flag to specify the node taint in the BGP LB node pool.

  Args:
   bare_metal_bgp_lb_node_config: The parent group to add the flags to.
  """
    bare_metal_bgp_lb_node_config.add_argument('--bgp-lb-load-balancer-node-taints', metavar='KEY=VALUE:EFFECT', help='Node taint applied to every node in a BGP LB node pool.', type=arg_parsers.ArgDict())