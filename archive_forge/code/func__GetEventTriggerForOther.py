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
def _GetEventTriggerForOther(args: parser_extensions.Namespace, messages: types.ModuleType) -> api_types.EventTrigger:
    """Constructs an EventTrigger when using `--trigger-[bucket|topic|filters]`.

  Args:
    args: arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.

  Returns:
    A `cloudfunctions_v2_messages.EventTrigger` used to request
      events sent from another service.
  """
    event_filters = []
    event_type = None
    pubsub_topic = None
    service_account_email = args.trigger_service_account or args.service_account
    trigger_location = args.trigger_location
    if args.trigger_topic:
        event_type = api_util.EA_PUBSUB_MESSAGE_PUBLISHED
        pubsub_topic = _BuildFullPubsubTopic(args.trigger_topic)
    elif args.trigger_bucket:
        bucket = args.trigger_bucket[5:].rstrip('/')
        event_type = api_util.EA_STORAGE_FINALIZE
        event_filters = [messages.EventFilter(attribute='bucket', value=bucket)]
    elif args.trigger_event_filters:
        event_type = args.trigger_event_filters.get('type')
        event_filters = [messages.EventFilter(attribute=attr, value=val) for attr, val in args.trigger_event_filters.items() if attr != 'type']
        if args.trigger_event_filters_path_pattern:
            operator = 'match-path-pattern'
            event_filters.extend([messages.EventFilter(attribute=attr, value=val, operator=operator) for attr, val in args.trigger_event_filters_path_pattern.items()])
    trigger_channel = None
    if args.trigger_channel:
        trigger_channel = args.CONCEPTS.trigger_channel.Parse().RelativeName()
    return messages.EventTrigger(eventFilters=event_filters, eventType=event_type, pubsubTopic=pubsub_topic, serviceAccountEmail=service_account_email, channel=trigger_channel, triggerRegion=trigger_location)