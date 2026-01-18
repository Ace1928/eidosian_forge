from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.printers import traffic_printer
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AlphaAdjustTraffic(AdjustTraffic):
    """Adjust the traffic assignments for a Cloud Run service."""

    @classmethod
    def Args(cls, parser):
        cls.CommonArgs(parser)
        managed_group = flags.GetManagedArgGroup(parser)
        flags.AddBinAuthzBreakglassFlag(managed_group)