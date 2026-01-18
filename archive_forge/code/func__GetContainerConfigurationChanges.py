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
def _GetContainerConfigurationChanges(container_args, container_name=None):
    """Returns per-container configuration changes."""
    changes = []
    if hasattr(container_args, 'image') and container_args.image is not None:
        changes.append(config_changes.ImageChange(container_args.image, container_name=container_name))
    if _HasEnvChanges(container_args):
        changes.append(_GetEnvChanges(container_args, container_name=container_name))
    if container_args.IsSpecified('cpu'):
        changes.append(config_changes.ResourceChanges(cpu=container_args.cpu, container_name=container_name))
    if container_args.IsSpecified('memory'):
        changes.append(config_changes.ResourceChanges(memory=container_args.memory, container_name=container_name))
    if container_args.IsKnownAndSpecified('gpu'):
        changes.append(config_changes.ResourceChanges(gpu=container_args.gpu, container_name=container_name))
    if container_args.IsSpecified('command'):
        changes.append(config_changes.ContainerCommandChange(container_args.command, container_name=container_name))
    if container_args.IsSpecified('args'):
        changes.append(config_changes.ContainerArgsChange(container_args.args, container_name=container_name))
    if FlagIsExplicitlySet(container_args, 'remove_volume_mount') or FlagIsExplicitlySet(container_args, 'clear_volume_mounts'):
        changes.append(config_changes.RemoveVolumeMountChange(removed_mounts=container_args.remove_volume_mount, clear_mounts=container_args.clear_volume_mounts, container_name=container_name))
    if _HasSecretsChanges(container_args):
        changes.extend(_GetSecretsChanges(container_args, container_name=container_name))
    if FlagIsExplicitlySet(container_args, 'add_volume_mount'):
        changes.append(config_changes.AddVolumeMountChange(new_mounts=container_args.add_volume_mount, container_name=container_name))
    return changes