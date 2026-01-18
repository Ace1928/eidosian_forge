from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddDynamicPortAllocationArgs(parser, for_create=False):
    """Adds arguments for Dynamic Port Allocation to specify the maximum number of ports per VM for NAT."""
    max_ports_help_text = textwrap.dedent('  Maximum ports to be allocated to a VM.\n\n  This field can only be set when Dynamic Port Allocation is enabled and\n  defaults to 65536. It must be set to a power of 2 that is greater than\n  minPortsPerVm and at most 65536.\n  ')
    _AddClearableArgument(parser, for_create, 'max-ports-per-vm', arg_parsers.BoundedInt(lower_bound=64, upper_bound=65536), max_ports_help_text, 'Clear maximum ports to be allocated to a VM')
    dpa_help_text = textwrap.dedent('  Enable dynamic port allocation.\n\n  If not specified, Dynamic Port Allocation is disabled by default.\n  ')
    parser.add_argument('--enable-dynamic-port-allocation', action=arg_parsers.StoreTrueFalseAction, help=dpa_help_text)