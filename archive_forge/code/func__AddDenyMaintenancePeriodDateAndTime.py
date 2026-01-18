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
def _AddDenyMaintenancePeriodDateAndTime(group, alloydb_messages):
    """Adds deny maintenance period start and end date and time flags to the group."""
    group.add_argument('--deny-maintenance-period-start-date', required=True, hidden=True, type=_GetDate(alloydb_messages), help='Date when the deny maintenance period begins, that is 2020-11-01 or 11-01 for recurring.')
    group.add_argument('--deny-maintenance-period-end-date', required=True, hidden=True, type=_GetDate(alloydb_messages), help='Date when the deny maintenance period ends, that is 2020-11-01 or 11-01 for recurring.')
    group.add_argument('--deny-maintenance-period-time', required=True, hidden=True, type=_GetTimeOfDay(alloydb_messages), help='Time when the deny maintenance period starts and ends, for example 05:00:00, in UTC time zone.')