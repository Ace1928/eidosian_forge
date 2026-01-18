from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
def _ActiveStatus(trigger):
    event_filters = trigger.get('eventFilters', trigger.get('matchingCriteria'))
    event_type = types.EventFiltersDictToType(event_filters)
    active_time = triggers.TriggerActiveTime(event_type, trigger['updateTime'])
    return 'By {}'.format(active_time) if active_time else 'Yes'