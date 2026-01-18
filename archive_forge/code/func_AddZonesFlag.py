from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def AddZonesFlag(parser):
    """Add flags for choosing zones for regional managed instance group."""
    parser.add_argument('--zones', metavar='ZONE', help='          If this flag is specified a regional managed instance group will be\n          created. The managed instance group will be in the same region as\n          specified zones and will spread instances in it between specified\n          zones.\n\n          All zones must belong to the same region. You may specify --region\n          flag but it must be the region to which zones belong. This flag is\n          mutually exclusive with --zone flag.', type=arg_parsers.ArgList(min_length=1), completer=compute_completers.ZonesCompleter, default=[])