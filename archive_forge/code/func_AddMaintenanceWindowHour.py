from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddMaintenanceWindowHour(parser, hidden=False):
    parser.add_argument('--maintenance-window-hour', type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=23), help='Hour of day for maintenance window, in UTC time zone.', hidden=hidden)