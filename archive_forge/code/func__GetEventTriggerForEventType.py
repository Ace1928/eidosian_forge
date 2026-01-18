import re
import types
from typing import FrozenSet, Optional, Tuple
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import types as api_types
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import labels_util
from googlecloudsdk.command_lib.functions import run_util
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
def _GetEventTriggerForEventType(args: parser_extensions.Namespace, messages: types.ModuleType) -> api_types.EventTrigger:
    """Constructs an EventTrigger message from the command-line arguments.

  Args:
    args: The arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.

  Returns:
    A `cloudfunctions_v2_messages.EventTrigger`, used to request
      events sent from another service.
  """
    trigger_event = args.trigger_event
    trigger_resource = args.trigger_resource
    service_account_email = args.trigger_service_account or args.service_account
    if trigger_event in api_util.PUBSUB_MESSAGE_PUBLISH_TYPES:
        pubsub_topic = api_util_v1.ValidatePubsubTopicNameOrRaise(trigger_resource)
        return messages.EventTrigger(eventType=api_util.EA_PUBSUB_MESSAGE_PUBLISHED, pubsubTopic=_BuildFullPubsubTopic(pubsub_topic), serviceAccountEmail=service_account_email, triggerRegion=args.trigger_location)
    elif trigger_event in api_util.EVENTARC_STORAGE_TYPES or trigger_event in api_util.EVENTFLOW_TO_EVENTARC_STORAGE_MAP:
        bucket_name = storage_util.BucketReference.FromUrl(trigger_resource).bucket
        storage_event_type = api_util.EVENTFLOW_TO_EVENTARC_STORAGE_MAP.get(trigger_event, trigger_event)
        return messages.EventTrigger(eventType=storage_event_type, eventFilters=[messages.EventFilter(attribute='bucket', value=bucket_name)], serviceAccountEmail=service_account_email, triggerRegion=args.trigger_location)
    else:
        raise exceptions.InvalidArgumentException('--trigger-event', 'Event type {} is not supported by this flag, try using --trigger-event-filters.'.format(trigger_event))