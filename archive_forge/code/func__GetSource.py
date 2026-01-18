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
def _GetSource(args: parser_extensions.Namespace, client: base_api.BaseApiClient, function_ref: resources.Resource, existing_function: Optional[api_types.Function]) -> Tuple[Optional[api_types.Source], FrozenSet[str]]:
    """Parses the source bucket and object from the --source flag.

  Args:
    args: arguments that this command was invoked with.
    client: The GCFv2 API client
    function_ref: The GCFv2 functions resource reference.
    existing_function: `cloudfunctions_v2_messages.Function | None`,
      pre-existing function.

  Returns:
    A tuple `(function_source, update_field_set)` where
    - `function_source` is the resulting `cloudfunctions_v2_messages.Source`,
    - `update_field_set` is a set of update mask fields.
  """
    if args.source is None and existing_function is not None and existing_function.buildConfig.source.repoSource:
        return (None, frozenset())
    source = args.source or '.'
    messages = client.MESSAGES_MODULE
    if source.startswith('gs://'):
        return (_GetSourceGCS(messages, source), frozenset(['build_config.source']))
    elif source.startswith('https://'):
        return (_GetSourceCSR(messages, source), frozenset(['build_config.source']))
    else:
        runtime = args.runtime or existing_function.buildConfig.runtime
        source_util.ValidateDirectoryHasRequiredRuntimeFiles(source, runtime)
        return (_GetSourceLocal(args, client, function_ref, source, kms_key=_GetActiveKmsKey(args, existing_function)), frozenset(['build_config.source']))