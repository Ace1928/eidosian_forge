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
def _GetEventTrigger(args: parser_extensions.Namespace, messages: types.ModuleType, existing_function: Optional[api_types.Function]) -> Tuple[Optional[api_types.EventTrigger], FrozenSet[str]]:
    """Constructs an EventTrigger message from the command-line arguments.

  Args:
    args: The arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.
    existing_function: The pre-existing function.

  Returns:
    A tuple `(event_trigger, update_fields_set)` where:
    - `event_trigger` is a `cloudfunctions_v2_messages.EventTrigger` used to
    request events sent from another service,
    - `updated_fields_set` is a set of update mask fields.
  """
    if args.trigger_http:
        event_trigger, updated_fields_set = (None, frozenset(['event_trigger'] if existing_function else []))
    elif args.trigger_event or args.trigger_resource:
        event_trigger, updated_fields_set = (_GetEventTriggerForEventType(args, messages), frozenset(['event_trigger']))
    elif args.trigger_topic or args.trigger_bucket or args.trigger_event_filters:
        event_trigger, updated_fields_set = (_GetEventTriggerForOther(args, messages), frozenset(['event_trigger']))
    elif existing_function:
        event_trigger, updated_fields_set = (existing_function.eventTrigger, frozenset())
    else:
        raise calliope_exceptions.OneOfArgumentsRequiredException(['--trigger-topic', '--trigger-bucket', '--trigger-http', '--trigger-event', '--trigger-event-filters'], 'You must specify a trigger when deploying a new function.')
    if args.IsSpecified('retry'):
        retry_policy, retry_updated_field = _GetRetry(args, messages, event_trigger)
        event_trigger.retryPolicy = retry_policy
        updated_fields_set = updated_fields_set.union(retry_updated_field)
    if event_trigger and trigger_types.IsPubsubType(event_trigger.eventType):
        deploy_util.ensure_pubsub_sa_has_token_creator_role()
    if event_trigger and trigger_types.IsAuditLogType(event_trigger.eventType):
        deploy_util.ensure_data_access_logs_are_enabled(event_trigger.eventFilters)
    return (event_trigger, updated_fields_set)