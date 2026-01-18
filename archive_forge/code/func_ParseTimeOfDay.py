from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseTimeOfDay(time_of_day, messages):
    hour_and_min = time_of_day.split(':')
    hour = int(hour_and_min[0])
    minute = int(hour_and_min[1])
    return messages.TimeOfDay(hours=hour, minutes=minute)