from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddNatNameArg(parser):
    """Adds an argument to identify the NAT to which the Rule belongs."""
    parser.add_argument('--nat', help='Name of the NAT that contains the Rule', required=True)