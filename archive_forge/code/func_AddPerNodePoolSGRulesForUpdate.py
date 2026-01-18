from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPerNodePoolSGRulesForUpdate(parser):
    """Adds --disable-per-node-pool-sg-rules and --enable-per-node-pool-sg-rules flags to parser."""
    group = parser.add_group('Default per node pool security group rules', mutex=True)
    AddPerNodePoolSGRules(group)
    group.add_argument('--enable-per-node-pool-sg-rules', action='store_false', default=None, dest='per_node_pool_sg_rules_disabled', help='Enable the default per node pool subnet security group rules on the control plane security group.')