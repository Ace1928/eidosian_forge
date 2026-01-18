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
def _GetBuildConfig(args: parser_extensions.Namespace, client: client_v2.FunctionsClient, function_ref: resources.Resource, existing_function: Optional[api_types.Function]) -> Tuple[api_types.BuildConfig, FrozenSet[str]]:
    """Constructs a BuildConfig message from the command-line arguments.

  Args:
    args: arguments that this command was invoked with.
    client: The GCFv2 API client.
    function_ref: The GCFv2 functions resource reference.
    existing_function: The pre-existing function.

  Returns:
    The resulting build config and the set of update mask fields.
  """
    function_source, source_updated_fields = _GetSource(args, client, function_ref, existing_function)
    old_build_env_vars = {}
    if existing_function and existing_function.buildConfig and existing_function.buildConfig.environmentVariables and existing_function.buildConfig.environmentVariables.additionalProperties:
        for additional_property in existing_function.buildConfig.environmentVariables.additionalProperties:
            old_build_env_vars[additional_property.key] = additional_property.value
    build_env_var_flags = map_util.GetMapFlagsFromArgs('build-env-vars', args)
    build_env_vars = map_util.ApplyMapFlags(old_build_env_vars, **build_env_var_flags)
    updated_fields = set()
    if build_env_vars != old_build_env_vars:
        updated_fields.add('build_config.environment_variables')
    if args.entry_point is not None:
        updated_fields.add('build_config.entry_point')
    if args.runtime is not None:
        updated_fields.add('build_config.runtime')
    worker_pool = None if args.clear_build_worker_pool else args.build_worker_pool
    if args.build_worker_pool is not None or args.clear_build_worker_pool:
        updated_fields.add('build_config.worker_pool')
    service_account = None
    if args.IsKnownAndSpecified('build_service_account'):
        updated_fields.add('build_config.service_account')
        service_account = args.build_service_account
    messages = client.MESSAGES_MODULE
    automatic_update_policy = None
    on_deploy_update_policy = None
    if args.IsSpecified('runtime_update_policy'):
        updated_fields.update(('build_config.automatic_update_policy', 'build_config.on_deploy_update_policy'))
        if args.runtime_update_policy == 'automatic':
            automatic_update_policy = messages.AutomaticUpdatePolicy()
        if args.runtime_update_policy == 'on-deploy':
            on_deploy_update_policy = messages.OnDeployUpdatePolicy()
    build_updated_fields = frozenset.union(source_updated_fields, updated_fields)
    return (messages.BuildConfig(entryPoint=args.entry_point, runtime=args.runtime, source=function_source, workerPool=worker_pool, environmentVariables=messages.BuildConfig.EnvironmentVariablesValue(additionalProperties=[messages.BuildConfig.EnvironmentVariablesValue.AdditionalProperty(key=key, value=value) for key, value in sorted(build_env_vars.items())]), serviceAccount=service_account, automaticUpdatePolicy=automatic_update_policy, onDeployUpdatePolicy=on_deploy_update_policy), build_updated_fields)