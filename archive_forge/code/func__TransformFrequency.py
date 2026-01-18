from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _TransformFrequency(resource):
    """Returns a string description of the patch deployment schedule."""
    if _ONE_TIME_SCHEDULE in resource:
        return 'Once: Scheduled for ' + resource[_ONE_TIME_SCHEDULE]['executeTime']
    elif _RECURRING_SCHEDULE in resource:
        output_format = 'Recurring - {} {}'
        schedule = resource[_RECURRING_SCHEDULE]
        if schedule['frequency'] == 'DAILY':
            return output_format.format('Daily', '')
        elif schedule['frequency'] == 'WEEKLY':
            return output_format.format('Weekly', '')
        elif schedule['frequency'] == 'MONTHLY':
            if schedule['monthly'].get('weekDayOfMonth', ''):
                return output_format.format('Monthly', 'on specific day(s)')
            else:
                return output_format.format('Monthly', 'on specific date(s)')
        else:
            return _DEFAULT_NO_VALUE
    else:
        return _DEFAULT_NO_VALUE