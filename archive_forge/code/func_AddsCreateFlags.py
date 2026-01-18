from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.cloudbuild import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddsCreateFlags(parser):
    parser.add_argument('--file', required=True, help='The YAML file to use as the PipelineRun/TaskRun configuration file.')
    AddsRegionResourceArg(parser)