from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddUpdateCloudRunDestinationArgs(parser, release_track, required=False):
    """Adds arguments related to trigger's Cloud Run fully-managed resource destination for update operations."""
    run_group = parser.add_group(required=required, help='Flags for updating a Cloud Run fully-managed resource destination.')
    resource_group = run_group.add_mutually_exclusive_group()
    AddDestinationRunServiceArg(resource_group)
    if release_track == base.ReleaseTrack.GA:
        AddDestinationRunJobArg(resource_group)
    AddDestinationRunRegionArg(run_group)
    destination_run_path_group = run_group.add_mutually_exclusive_group()
    AddDestinationRunPathArg(destination_run_path_group)
    AddClearDestinationRunPathArg(destination_run_path_group)