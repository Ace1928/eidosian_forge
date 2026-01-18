from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetPipelineTypeArg(required=True):
    choices = {'batch': 'Specifies a Batch pipeline.', 'streaming': 'Specifies a Streaming pipeline.'}
    return base.ChoiceArgument('--pipeline-type', choices=choices, required=required, help_str="Type of the pipeline. One of 'BATCH' or 'STREAMING'.")