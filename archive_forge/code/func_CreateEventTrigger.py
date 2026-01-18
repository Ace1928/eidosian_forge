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
def CreateEventTrigger(trigger_provider, trigger_event, trigger_resource):
    """Create event trigger message.

  Args:
    trigger_provider: str, trigger provider label.
    trigger_event: str, trigger event label.
    trigger_resource: str, trigger resource name.

  Returns:
    A EventTrigger protobuf message.
  """
    messages = api_util.GetApiMessagesModule()
    event_trigger = messages.EventTrigger()
    event_trigger.eventType = trigger_event
    if trigger_provider == triggers.UNADVERTISED_PROVIDER_LABEL:
        event_trigger.resource = trigger_resource
    else:
        event_trigger.resource = ConvertTriggerArgsToRelativeName(trigger_provider, trigger_event, trigger_resource)
    return event_trigger