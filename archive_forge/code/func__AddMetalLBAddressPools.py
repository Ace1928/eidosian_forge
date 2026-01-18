from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddMetalLBAddressPools(metal_lb_config_group, is_update=False):
    """Adds flags for address pools used by Metal LB load balancer.

  Args:
    metal_lb_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    metal_lb_address_pools_mutex_group = metal_lb_config_group.add_group(help='MetalLB address pools configuration.', mutex=True, required=required)
    metal_lb_address_pools_from_file_help_text = '\nPath of the YAML/JSON file that contains the MetalLB address pools.\n\nExamples:\n\n  addressPools:\n  - pool: pool-1\n    addresses:\n    - 10.200.0.14/32\n    - 10.200.0.15/32\n    avoidBuggyIPs: True\n    manualAssign: True\n  - pool: pool-2\n    addresses:\n    - 10.200.0.16/32\n    avoidBuggyIPs: False\n    manualAssign: False\n\nList of supported fields in `addressPools`\n\nKEY           | VALUE                 | NOTE\n--------------|-----------------------|---------------------------\npool          | string                | required, mutable\naddresses     | one or more IP ranges | required, mutable\navoidBuggyIPs | bool                  | optional, mutable, defaults to False\nmanualAssign  | bool                  | optional, mutable, defaults to False\n\n'
    metal_lb_address_pools_mutex_group.add_argument('--metal-lb-address-pools-from-file', help=metal_lb_address_pools_from_file_help_text, type=arg_parsers.YAMLFileContents(), hidden=True)
    metal_lb_address_pools_mutex_group.add_argument('--metal-lb-address-pools', help='MetalLB address pools configuration.', action='append', type=arg_parsers.ArgDict(spec={'pool': str, 'avoid-buggy-ips': arg_parsers.ArgBoolean(), 'manual-assign': arg_parsers.ArgBoolean(), 'addresses': arg_parsers.ArgList(custom_delim_char=';')}, required_keys=['pool', 'addresses']))