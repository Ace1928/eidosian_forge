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
def GetWorkerConfigurationChanges(args, release_track=base.ReleaseTrack.ALPHA, for_update=False):
    """Returns a list of changes to the worker config, based on the flags set."""
    changes = []
    if not for_update:
        changes.append(config_changes.SetAnnotationChange(service.INGRESS_ANNOTATION, 'none'))
        changes.append(config_changes.CpuThrottlingChange(throttling=False))
        changes.append(config_changes.HealthCheckChange(health_check=False))
        changes.append(config_changes.DefaultUrlChange(default_url=False))
        changes.append(config_changes.SandboxChange('gen2'))
    changes.extend(_GetConfigurationChanges(args, release_track=release_track))
    changes.extend(_GetWorkerScalingChanges(args))
    if _HasInstanceSplitChanges(args):
        changes.append(_GetInstanceSplitChanges(args))
    if 'no_promote' in args and args.no_promote:
        changes.append(config_changes.NoPromoteChange())
    if 'update_annotations' in args and args.update_annotations:
        for key, value in args.update_annotations.items():
            changes.append(config_changes.SetAnnotationChange(key, value))
    if 'revision_suffix' in args and args.revision_suffix:
        changes.append(config_changes.RevisionNameChanges(args.revision_suffix))
    if 'gpu_type' in args and args.gpu_type:
        changes.append(config_changes.GpuTypeChange(gpu_type=args.gpu_type))
    _PrependClientNameAndVersionChange(args, changes)
    if FlagIsExplicitlySet(args, 'depends_on'):
        changes.append(config_changes.ContainerDependenciesChange({'': args.depends_on}))
    if FlagIsExplicitlySet(args, 'containers'):
        dependency_changes = {container_name: container_args.depends_on for container_name, container_args in args.containers.items() if container_args.IsSpecified('depends_on')}
        if dependency_changes:
            changes.append(config_changes.ContainerDependenciesChange(dependency_changes))
    return changes