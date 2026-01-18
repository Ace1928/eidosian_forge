from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import os.path
from googlecloudsdk.api_lib.run import api_enabler
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import container_parser
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def ContainerArgGroup():
    """Returns an argument group with all per-container deploy args."""
    help_text = '\nContainer Flags\n\n  If the --container or --remove-containers flag is specified the following\n  arguments may only be specified after a --container flag.\n'
    group = base.ArgumentGroup(help=help_text)
    group.AddArgument(flags.SourceAndImageFlags(image='us-docker.pkg.dev/cloudrun/container/job:latest'))
    group.AddArgument(flags.MutexEnvVarsFlags())
    group.AddArgument(flags.MemoryFlag())
    group.AddArgument(flags.CpuFlag())
    group.AddArgument(flags.ArgsFlag())
    group.AddArgument(flags.SecretsFlags())
    group.AddArgument(flags.CommandFlag())
    group.AddArgument(flags.DependsOnFlag())
    group.AddArgument(flags.AddVolumeMountFlag())
    group.AddArgument(flags.RemoveVolumeMountFlag())
    group.AddArgument(flags.ClearVolumeMountsFlag())
    return group