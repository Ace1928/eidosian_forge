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
def GetServiceConfigurationChanges(args, release_track=base.ReleaseTrack.GA):
    """Returns a list of changes to the service config, based on the flags set."""
    changes = _GetConfigurationChanges(args, release_track=release_track)
    changes.extend(_GetScalingChanges(args))
    changes.extend(_GetServiceScalingChanges(args))
    if _HasTrafficChanges(args):
        changes.append(_GetTrafficChanges(args))
    if 'no_traffic' in args and args.no_traffic:
        changes.append(config_changes.NoTrafficChange())
    if 'concurrency' in args and args.concurrency:
        changes.append(config_changes.ConcurrencyChanges.FromFlag(args.concurrency))
    if 'timeout' in args and args.timeout:
        changes.append(config_changes.TimeoutChanges(timeout=args.timeout))
    if 'update_annotations' in args and args.update_annotations:
        for key, value in args.update_annotations.items():
            changes.append(config_changes.SetAnnotationChange(key, value))
    if 'revision_suffix' in args and args.revision_suffix:
        changes.append(config_changes.RevisionNameChanges(args.revision_suffix))
    if 'connectivity' in args and args.connectivity:
        if args.connectivity == 'internal':
            changes.append(config_changes.EndpointVisibilityChange(True))
        elif args.connectivity == 'external':
            changes.append(config_changes.EndpointVisibilityChange(False))
    if FlagIsExplicitlySet(args, 'ingress'):
        changes.append(_GetIngressChanges(args))
    if FlagIsExplicitlySet(args, 'port'):
        changes.append(config_changes.ContainerPortChange(port=args.port))
    if FlagIsExplicitlySet(args, 'use_http2'):
        changes.append(config_changes.ContainerPortChange(use_http2=args.use_http2))
    if FlagIsExplicitlySet(args, 'tag'):
        changes.append(config_changes.TagOnDeployChange(args.tag))
    if FlagIsExplicitlySet(args, 'cpu_throttling'):
        changes.append(config_changes.CpuThrottlingChange(throttling=args.cpu_throttling))
    if FlagIsExplicitlySet(args, 'cpu_boost'):
        changes.append(config_changes.StartupCpuBoostChange(cpu_boost=args.cpu_boost))
    if FlagIsExplicitlySet(args, 'deploy_health_check'):
        changes.append(config_changes.HealthCheckChange(health_check=args.deploy_health_check))
    if FlagIsExplicitlySet(args, 'default_url'):
        changes.append(config_changes.DefaultUrlChange(default_url=args.default_url))
    if FlagIsExplicitlySet(args, 'invoker_iam_check'):
        changes.append(config_changes.InvokerIamChange(invoker_iam_check=args.invoker_iam_check))
    if FlagIsExplicitlySet(args, 'session_affinity'):
        if args.session_affinity:
            changes.append(config_changes.SetTemplateAnnotationChange(revision.SESSION_AFFINITY_ANNOTATION, str(args.session_affinity).lower()))
        else:
            changes.append(config_changes.DeleteTemplateAnnotationChange(revision.SESSION_AFFINITY_ANNOTATION))
    if FlagIsExplicitlySet(args, 'runtime'):
        changes.append(config_changes.RuntimeChange(runtime=args.runtime))
    if 'gpu_type' in args and args.gpu_type:
        changes.append(config_changes.GpuTypeChange(gpu_type=args.gpu_type))
    _PrependClientNameAndVersionChange(args, changes)
    if FlagIsExplicitlySet(args, 'depends_on'):
        changes.append(config_changes.ContainerDependenciesChange({'': args.depends_on}))
    if FlagIsExplicitlySet(args, 'containers'):
        for container_name, container_args in args.containers.items():
            changes.extend(_GetServiceContainerChanges(container_args, container_name))
        dependency_changes = {container_name: container_args.depends_on for container_name, container_args in args.containers.items() if container_args.IsSpecified('depends_on')}
        if dependency_changes:
            changes.append(config_changes.ContainerDependenciesChange(dependency_changes))
        base_image_changes = _GetBaseImageChanges(args)
        if base_image_changes:
            changes.extend(base_image_changes)
    if FlagIsExplicitlySet(args, 'domain'):
        changes.append(config_changes.MultiRegionDomainNameChange(domain_name=args.domain))
    return changes