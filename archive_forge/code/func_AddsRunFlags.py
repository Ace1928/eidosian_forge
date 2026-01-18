from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.cloudbuild import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddsRunFlags(parser):
    """Add flags related to a run to parser."""
    parser.add_argument('RUN_ID', help='The ID of the PipelineRun/TaskRun.')
    parser.add_argument('--type', choices=['pipelinerun', 'taskrun'], default='pipelinerun', help='Type of Run.')
    AddsRegionResourceArg(parser)