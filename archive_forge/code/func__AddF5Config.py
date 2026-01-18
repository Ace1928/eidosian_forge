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
def _AddF5Config(lb_config_mutex_group, for_update=False):
    """Adds flags for F5 Big IP load balancer.

  Args:
    lb_config_mutex_group: The parent mutex group to add the flags to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    if for_update:
        return
    f5_config_group = lb_config_mutex_group.add_group(help='F5 Big IP Configuration')
    f5_config_group.add_argument('--f5-config-address', type=str, required=required, help='F5 Big IP load balancer address.')
    f5_config_group.add_argument('--f5-config-partition', type=str, required=required, help='F5 Big IP load balancer partition.')
    f5_config_group.add_argument('--f5-config-snat-pool', type=str, help='F5 Big IP load balancer pool name if using SNAT.')