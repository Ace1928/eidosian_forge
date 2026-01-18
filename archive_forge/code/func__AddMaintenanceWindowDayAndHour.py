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
def _AddMaintenanceWindowDayAndHour(group, alloydb_messages):
    """Adds maintenance window day and hour flags to the group."""
    day_of_week_enum = alloydb_messages.MaintenanceWindow.DayValueValuesEnum
    group.add_argument('--maintenance-window-day', required=True, hidden=True, type=day_of_week_enum, choices=[day_of_week_enum.MONDAY, day_of_week_enum.TUESDAY, day_of_week_enum.WEDNESDAY, day_of_week_enum.THURSDAY, day_of_week_enum.FRIDAY, day_of_week_enum.SATURDAY, day_of_week_enum.SUNDAY], help='Day of week for maintenance window, in UTC time zone.')
    group.add_argument('--maintenance-window-hour', required=True, hidden=True, type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=23), help='Hour of day for maintenance window, in UTC time zone.')