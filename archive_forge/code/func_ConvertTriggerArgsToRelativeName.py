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
def ConvertTriggerArgsToRelativeName(trigger_provider, trigger_event, trigger_resource):
    """Prepares resource field for Function EventTrigger to use in API call.

  API uses relative resource name in EventTrigger message field. The
  structure of that identifier depends on the resource type which depends on
  combination of --trigger-provider and --trigger-event arguments' values.
  This function chooses the appropriate form, fills it with required data and
  returns as a string.

  Args:
    trigger_provider: The --trigger-provider flag value.
    trigger_event: The --trigger-event flag value.
    trigger_resource: The --trigger-resource flag value.

  Returns:
    Relative resource name to use in EventTrigger field.
  """
    resource_type = triggers.TRIGGER_PROVIDER_REGISTRY.Event(trigger_provider, trigger_event).resource_type
    params = {}
    if resource_type.value.collection_id in {'google.firebase.analytics.event', 'google.firebase.database.ref', 'google.firestore.document'}:
        return trigger_resource
    elif resource_type.value.collection_id == 'cloudresourcemanager.projects':
        params['projectId'] = properties.VALUES.core.project.GetOrFail
    elif resource_type.value.collection_id == 'pubsub.projects.topics':
        params['projectsId'] = properties.VALUES.core.project.GetOrFail
    elif resource_type.value.collection_id == 'cloudfunctions.projects.buckets':
        pass
    ref = resources.REGISTRY.Parse(trigger_resource, params, collection=resource_type.value.collection_id)
    return ref.RelativeName()