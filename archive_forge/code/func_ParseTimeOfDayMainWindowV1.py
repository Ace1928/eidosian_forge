from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseTimeOfDayMainWindowV1(time_of_day):
    """Convert input to TimeOfDay type for Main Window v1."""
    messages = GetMessagesModuleForVersion('v1')
    arg = '--maintenance-window-time'
    error_message = "'--maintenance-window-time' must be used in a valid 24-hr UTC Time format."
    CheckTimeOfDayField(time_of_day, error_message, arg)
    return ParseTimeOfDay(time_of_day, messages)