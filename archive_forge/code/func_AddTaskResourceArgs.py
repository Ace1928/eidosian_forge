from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddTaskResourceArgs(parser):
    """Add the task resource argument.

  Args:
    parser: the parser for the command.
  """
    arg_specs = [presentation_specs.ResourcePresentationSpec('TASK', GetTaskResourceSpec(), 'The Batch task resource. If not specified,the current batch/location is used.', required=True)]
    concept_parsers.ConceptParser(arg_specs).AddToParser(parser)