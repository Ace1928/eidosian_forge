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
def GetJobConfigurationChanges(args, release_track=base.ReleaseTrack.GA):
    """Returns a list of changes to the job config, based on the flags set."""
    changes = _GetConfigurationChanges(args, release_track=release_track)
    changes.insert(0, config_changes.DeleteAnnotationChange(k8s_object.BINAUTHZ_BREAKGLASS_ANNOTATION))
    if FlagIsExplicitlySet(args, 'parallelism'):
        changes.append(config_changes.ExecutionTemplateSpecChange('parallelism', args.parallelism))
    if FlagIsExplicitlySet(args, 'tasks'):
        changes.append(config_changes.ExecutionTemplateSpecChange('taskCount', args.tasks))
    if FlagIsExplicitlySet(args, 'image'):
        changes.append(config_changes.JobNonceChange())
    if FlagIsExplicitlySet(args, 'max_retries'):
        changes.append(config_changes.JobMaxRetriesChange(args.max_retries))
    if FlagIsExplicitlySet(args, 'task_timeout'):
        changes.append(config_changes.JobTaskTimeoutChange(args.task_timeout))
    _PrependClientNameAndVersionChange(args, changes)
    if FlagIsExplicitlySet(args, 'containers'):
        dependency_changes = {container_name: container_args.depends_on for container_name, container_args in args.containers.items() if container_args.IsSpecified('depends_on')}
        if dependency_changes:
            changes.append(config_changes.ContainerDependenciesChange(dependency_changes))
    return changes