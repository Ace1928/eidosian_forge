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
def _GetServiceConfig(args: parser_extensions.Namespace, messages: types.ModuleType, existing_function: Optional[api_types.Function]) -> Tuple[api_types.ServiceConfig, FrozenSet[str]]:
    """Constructs a ServiceConfig message from the command-line arguments.

  Args:
    args: arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.
    existing_function: the existing function.

  Returns:
    A tuple `(service_config, updated_fields_set)` where
    - `service_config` is the resulting
    `cloudfunctions_v2_messages.ServiceConfig`.
    - `updated_fields_set` is a set of update mask fields.
  """
    old_env_vars = {}
    if existing_function and existing_function.serviceConfig and existing_function.serviceConfig.environmentVariables and existing_function.serviceConfig.environmentVariables.additionalProperties:
        for additional_property in existing_function.serviceConfig.environmentVariables.additionalProperties:
            old_env_vars[additional_property.key] = additional_property.value
    env_var_flags = map_util.GetMapFlagsFromArgs('env-vars', args)
    env_vars = map_util.ApplyMapFlags(old_env_vars, **env_var_flags)
    old_secrets = {}
    new_secrets = {}
    if existing_function and existing_function.serviceConfig:
        old_secrets = secrets_util.GetSecretsAsDict(existing_function.serviceConfig.secretEnvironmentVariables, existing_function.serviceConfig.secretVolumes)
    if secrets_config.IsArgsSpecified(args):
        try:
            new_secrets = secrets_config.ApplyFlags(old_secrets, args, api_util.GetProject(), projects_util.GetProjectNumber(api_util.GetProject()))
        except ArgumentTypeError as error:
            core_exceptions.reraise(exceptions.FunctionsError(error))
    else:
        new_secrets = old_secrets
    old_secret_env_vars, old_secret_volumes = secrets_config.SplitSecretsDict(old_secrets)
    secret_env_vars, secret_volumes = secrets_config.SplitSecretsDict(new_secrets)
    vpc_connector, vpc_egress_settings, vpc_updated_fields = _GetVpcAndVpcEgressSettings(args, messages, existing_function)
    ingress_settings, ingress_updated_fields = _GetIngressSettings(args, messages)
    concurrency = getattr(args, 'concurrency', None)
    cpu = getattr(args, 'cpu', None)
    updated_fields = set()
    if args.serve_all_traffic_latest_revision:
        updated_fields.add('service_config.all_traffic_on_latest_revision')
    if args.memory is not None:
        updated_fields.add('service_config.available_memory')
    if concurrency is not None:
        updated_fields.add('service_config.max_instance_request_concurrency')
    if cpu is not None:
        updated_fields.add('service_config.available_cpu')
    if args.max_instances is not None or args.clear_max_instances:
        updated_fields.add('service_config.max_instance_count')
    if args.min_instances is not None or args.clear_min_instances:
        updated_fields.add('service_config.min_instance_count')
    if args.run_service_account is not None or args.service_account is not None:
        updated_fields.add('service_config.service_account_email')
    if args.timeout is not None:
        updated_fields.add('service_config.timeout_seconds')
    if env_vars != old_env_vars:
        updated_fields.add('service_config.environment_variables')
    if secret_env_vars != old_secret_env_vars:
        updated_fields.add('service_config.secret_environment_variables')
    if secret_volumes != old_secret_volumes:
        updated_fields.add('service_config.secret_volumes')
    service_updated_fields = frozenset.union(vpc_updated_fields, ingress_updated_fields, updated_fields)
    return (messages.ServiceConfig(availableMemory=_ParseMemoryStrToK8sMemory(args.memory), maxInstanceCount=None if args.clear_max_instances else args.max_instances, minInstanceCount=None if args.clear_min_instances else args.min_instances, serviceAccountEmail=args.run_service_account or args.service_account, timeoutSeconds=args.timeout, ingressSettings=ingress_settings, vpcConnector=vpc_connector, vpcConnectorEgressSettings=vpc_egress_settings, allTrafficOnLatestRevision=args.serve_all_traffic_latest_revision or None, environmentVariables=messages.ServiceConfig.EnvironmentVariablesValue(additionalProperties=[messages.ServiceConfig.EnvironmentVariablesValue.AdditionalProperty(key=key, value=value) for key, value in sorted(env_vars.items())]), secretEnvironmentVariables=secrets_util.SecretEnvVarsToMessages(secret_env_vars, messages), secretVolumes=secrets_util.SecretVolumesToMessages(secret_volumes, messages, normalize_for_v2=True), maxInstanceRequestConcurrency=concurrency, availableCpu=_ValidateK8sCpuStr(cpu)), service_updated_fields)