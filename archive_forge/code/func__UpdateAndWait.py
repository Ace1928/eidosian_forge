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
def _UpdateAndWait(gcf_client: client_v2.FunctionsClient, function_ref: resources.Resource, function: api_types.Function, updated_fields_set: FrozenSet[str]) -> None:
    """Update a function.

  This does not include setting the invoker permissions.

  Args:
    gcf_client: The GCFv2 API client.
    function_ref: The GCFv2 functions resource reference.
    function: `cloudfunctions_v2_messages.Function`, The function to update.
    updated_fields_set: A set of update mask fields.

  Returns:
    None
  """
    client = gcf_client.client
    messages = gcf_client.messages
    if updated_fields_set:
        update_request = messages.CloudfunctionsProjectsLocationsFunctionsPatchRequest(name=function_ref.RelativeName(), updateMask=','.join(sorted(updated_fields_set)), function=function)
        operation = client.projects_locations_functions.Patch(update_request)
        operation_description = 'Updating function (may take a while)'
        api_util.WaitForOperation(client, messages, operation, operation_description, _EXTRA_STAGES)
    else:
        log.status.Print('Nothing to update.')