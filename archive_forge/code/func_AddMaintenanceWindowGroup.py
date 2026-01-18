from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddMaintenanceWindowGroup(parser, hidden=False, recurring_windows_hidden=False):
    """Adds a mutex for --maintenance-window and --maintenance-window-*."""
    maintenance_group = parser.add_group(hidden=hidden, mutex=True)
    maintenance_group.help = 'One of either maintenance-window or the group of maintenance-window flags can\nbe set.\n'
    AddDailyMaintenanceWindowFlag(maintenance_group)
    AddRecurringMaintenanceWindowFlags(maintenance_group, hidden=recurring_windows_hidden)