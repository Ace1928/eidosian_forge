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
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AlphaDeploy(BetaDeploy):
    """Create or update a Cloud Run job."""

    @classmethod
    def Args(cls, parser):
        Deploy.CommonArgs(parser, add_container_args=False)
        container_args = ContainerArgGroup()
        container_parser.AddContainerFlags(parser, container_args)
        flags.AddVpcNetworkGroupFlagsForUpdate(parser, resource_kind='job')
        flags.AddVolumesFlags(parser, cls.ReleaseTrack())
        flags.RemoveContainersFlag().AddToParser(parser)