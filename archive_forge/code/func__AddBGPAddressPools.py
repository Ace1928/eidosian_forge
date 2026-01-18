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
def _AddBGPAddressPools(bgp_lb_config_group, is_update=False):
    """Adds a flag for BGP address pool field.

  Args:
    bgp_lb_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    bgp_address_pools_help_text = "\nBGP load balancer address pools configurations.\n\nExamples:\n\nTo specify configurations for two address pools `pool1` and `pool2`,\n\n```\n$ {command} example_cluster\n      --bgp-address-pools 'pool=pool1,avoid-buggy-ips=True,manual-assign=True,addresses=192.168.1.1/32;192.168.1.2-192.168.1.3'\n      --bgp-address-pools 'pool=pool2,avoid-buggy-ips=False,manual-assign=False,addresses=192.168.2.1/32;192.168.2.2-192.168.2.3'\n```\n\nUse quote around the flag value to escape semicolon in the terminal.\n"
    required = not is_update
    bgp_lb_config_group.add_argument('--bgp-address-pools', help=bgp_address_pools_help_text, action='append', required=required, type=arg_parsers.ArgDict(spec={'pool': str, 'avoid-buggy-ips': arg_parsers.ArgBoolean(), 'manual-assign': arg_parsers.ArgBoolean(), 'addresses': arg_parsers.ArgList(custom_delim_char=';')}, required_keys=['pool', 'addresses']))