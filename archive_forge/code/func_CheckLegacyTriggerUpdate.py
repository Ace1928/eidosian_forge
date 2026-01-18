from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import triggers
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def CheckLegacyTriggerUpdate(function_trigger, new_trigger_event):
    if function_trigger:
        function_event_type = function_trigger.eventType
        if function_event_type in LEGACY_TRIGGER_EVENTS and function_event_type != new_trigger_event:
            error = LEGACY_TRIGGER_EVENTS[function_event_type]
            raise TriggerCompatibilityError(error)