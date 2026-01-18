from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddAdminManualLBConfig(bare_metal_admin_load_balancer_config_group):
    """Adds flags for manual load balancer.

  Args:
    bare_metal_admin_load_balancer_config_group: The parent group to add the
      flags to.
  """
    manual_lb_config_group = bare_metal_admin_load_balancer_config_group.add_group(help='Manual load balancer configuration.')
    manual_lb_config_group.add_argument('--enable-manual-lb', required=True, action='store_true', help='ManualLB typed load balancers configuration.')