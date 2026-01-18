from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEndpointTypesArg(parser):
    """Adds the --endpoint-type argument."""
    help_text = textwrap.dedent('    Endpoint Types supported by the NAT Gateway.\n\n    ENDPOINT_TYPE must be one of:\n\n    ENDPOINT_TYPE_VM\n      For VM Endpoints\n    ENDPOINT_TYPE_SWG\n      For Secure Web Gateway Endpoints\n    ENDPOINT_TYPE_MANAGED_PROXY_LB\n      For regional Application Load Balancers (internal and external) and regional proxy Network Load Balancers (internal and external) endpoints\n\n  The default is ENDPOINT_TYPE_VM.\n  ')
    choices = ['ENDPOINT_TYPE_VM', 'ENDPOINT_TYPE_SWG', 'ENDPOINT_TYPE_MANAGED_PROXY_LB']
    parser.add_argument('--endpoint-types', type=arg_parsers.ArgList(choices=choices), help=help_text, metavar='ENDPOINT_TYPE', required=False)