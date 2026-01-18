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
def ValidateTriggerArgs(trigger_event, trigger_resource, retry_specified, trigger_http_specified):
    """Check if args related function triggers are valid.

  Args:
    trigger_event: The trigger event
    trigger_resource: The trigger resource
    retry_specified: Whether or not `--retry` was specified
    trigger_http_specified: Whether or not `--trigger-http` was specified

  Raises:
    FunctionsError.
  """
    trigger_provider = triggers.TRIGGER_PROVIDER_REGISTRY.ProviderForEvent(trigger_event)
    trigger_provider_label = trigger_provider.label
    if trigger_provider_label != triggers.UNADVERTISED_PROVIDER_LABEL:
        resource_type = triggers.TRIGGER_PROVIDER_REGISTRY.Event(trigger_provider_label, trigger_event).resource_type
        if trigger_resource is None and resource_type != triggers.Resources.PROJECT:
            raise exceptions.FunctionsError('You must provide --trigger-resource when using --trigger-event={}'.format(trigger_event))
    if retry_specified and trigger_http_specified:
        raise calliope_exceptions.ConflictingArgumentsException('--trigger-http', '--retry')