import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def VerifyGKEFlags(args, release_track, product):
    """Raise ConfigurationError if args includes OnePlatform only arguments."""
    error_msg = 'The `{flag}` flag is not supported with Cloud Run for Anthos deployed on Google Cloud. Specify `--platform {platform}` or run `gcloud config set run/platform {platform}` to work with {platform_desc}.'
    if FlagIsExplicitlySet(args, 'allow_unauthenticated'):
        raise serverless_exceptions.ConfigurationError('The `--[no-]allow-unauthenticated` flag is not supported with Cloud Run for Anthos deployed on Google Cloud. All deployed services allow unauthenticated requests. The `--connectivity` flag can limit which network a service is available on to reduce access.')
    if FlagIsExplicitlySet(args, 'connectivity') and FlagIsExplicitlySet(args, 'ingress'):
        raise serverless_exceptions.ConfigurationError('Cannot specify both the `--connectivity` and `--ingress` flags. `--connectivity` is deprecated in favor of `--ingress`.')
    if FlagIsExplicitlySet(args, 'region'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--region', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'execution_environment'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--execution-environment', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'vpc_connector'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--vpc-connector', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_vpc_connector'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-vpc-connector', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'vpc_egress'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--vpc-egress', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'binary_authorization'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--binary-authorization', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_binary_authorization'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-binary-authorization', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'breakglass'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--breakglass', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'network'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--network', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'subnet'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--subnet', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'network-tags'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--network-tags', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'key'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--key', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'post_key_revocation_action_type'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--post-key-revocation-action-type', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'encryption_key_shutdown_hours'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--encryption-key-shutdown-hours', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_key'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-key', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_post_key_revocation_action_type'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-post-key-revocation-action-type', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_encryption_key_shutdown_hours'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-encryption-key-shutdown-hours', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'set_custom_audiences'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--set-custom-audiences', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'add_custom_audiences'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--add-custom-audiences', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'remove_custom_audiences'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--remove-custom-audiences', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'clear_custom_audiences'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--clear-custom-audiences', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'session_affinity'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--session-affinity', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))
    if FlagIsExplicitlySet(args, 'kubeconfig'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--kubeconfig', platform=platforms.PLATFORM_KUBERNETES, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_KUBERNETES]))
    if FlagIsExplicitlySet(args, 'context'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--context', platform=platforms.PLATFORM_KUBERNETES, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_KUBERNETES]))
    if FlagIsExplicitlySet(args, 'add_volume'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--add-volume', platform=platforms.PLATFORM_MANAGED, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_MANAGED]))