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
def _GetEventTriggerEventParams(trigger_event, trigger_resource):
    """Get the args for creating an event trigger.

  Args:
    trigger_event: The trigger event
    trigger_resource: The trigger resource

  Returns:
    A dictionary containing trigger_provider, trigger_event, and
    trigger_resource.
  """
    trigger_provider = triggers.TRIGGER_PROVIDER_REGISTRY.ProviderForEvent(trigger_event)
    trigger_provider_label = trigger_provider.label
    result = {'trigger_provider': trigger_provider_label, 'trigger_event': trigger_event, 'trigger_resource': trigger_resource}
    if trigger_provider_label == triggers.UNADVERTISED_PROVIDER_LABEL:
        return result
    resource_type = triggers.TRIGGER_PROVIDER_REGISTRY.Event(trigger_provider_label, trigger_event).resource_type
    if resource_type == triggers.Resources.TOPIC:
        trigger_resource = api_util.ValidatePubsubTopicNameOrRaise(trigger_resource)
    elif resource_type == triggers.Resources.BUCKET:
        trigger_resource = storage_util.BucketReference.FromUrl(trigger_resource).bucket
    elif resource_type in [triggers.Resources.FIREBASE_ANALYTICS_EVENT, triggers.Resources.FIREBASE_DB, triggers.Resources.FIRESTORE_DOC]:
        pass
    elif resource_type == triggers.Resources.PROJECT:
        if trigger_resource:
            properties.VALUES.core.project.Validate(trigger_resource)
    else:
        raise core_exceptions.InternalError()
    result['trigger_resource'] = trigger_resource
    return result