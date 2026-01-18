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
def VerifyManagedFlags(args, release_track, product):
    """Raise ConfigurationError if args aren't valid for managed Cloud Run."""
    if product == Product.EVENTS:
        raise serverless_exceptions.ConfigurationError('The flag --platform={0} is not supported. Instead of using the flag --platform={0} in "gcloud events", run "gcloud eventarc".'.format(platforms.PLATFORM_MANAGED))
    error_msg = 'The `{flag}` flag is not supported on the fully managed version of Cloud Run. Specify `--platform {platform}` or run `gcloud config set run/platform {platform}` to work with {platform_desc}.'
    if FlagIsExplicitlySet(args, 'connectivity'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--connectivity=[internal|external]', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'namespace'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--namespace', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'cluster'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--cluster', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'cluster_location'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--cluster-location', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if _HasConfigMapsChanges(args):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--[update|set|remove|clear]-config-maps', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'broker'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--broker', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'custom_type') and product == Product.EVENTS:
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--custom-type', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))
    if FlagIsExplicitlySet(args, 'kubeconfig'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--kubeconfig', platform=platforms.PLATFORM_KUBERNETES, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_KUBERNETES]))
    if FlagIsExplicitlySet(args, 'context'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--context', platform=platforms.PLATFORM_KUBERNETES, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_KUBERNETES]))
    if FlagIsExplicitlySet(args, 'trigger_filters'):
        raise serverless_exceptions.ConfigurationError(error_msg.format(flag='--trigger-filters', platform=platforms.PLATFORM_GKE, platform_desc=platforms.PLATFORM_SHORT_DESCRIPTIONS[platforms.PLATFORM_GKE]))