from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMaintenanceWindowArgToParser(parser):
    """Add flag for adding maintenance window start time to node group."""
    parser.add_argument('--maintenance-window-start-time', metavar='START_TIME', help='The time (in GMT) when planned maintenance operations window begins. The possible values are 00:00, 04:00, 08:00, 12:00, 16:00, 20:00.')