from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAssignInboundPublicIp(parser, update=False):
    """Adds Assign Inbound Public IP flag.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    update: If False, only allows the user to disable public IP.
  """
    if update:
        parser.add_argument('--assign-inbound-public-ip', required=False, type=str, help='Specify to enable or disable public IP on an instance.\n            ASSIGN_INBOUND_PUBLIC_IP must be one of:\n            * *NO_PUBLIC_IP*\n            ** This disables public IP on the instance. Updating an instance to\n            disable public IP will clear the list of authorized networks.\n            * *ASSIGN_IPV4*\n            ** Assign an inbound public IPv4 address for the instance.\n            public IP is enabled.')
    else:
        parser.add_argument('--assign-inbound-public-ip', required=False, type=str, choices={'NO_PUBLIC_IP': 'This disables public IP on the instance.'}, help='Specify to enable or disable public IP on an instance. On instance creation only disabling public IP is allowed. If you want to enable public IP, an instance must be created with public IP disabled first, then update the instance to enable public IP.')