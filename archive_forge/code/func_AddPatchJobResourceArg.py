from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPatchJobResourceArg(parser, verb, plural=False):
    """Creates a resource argument for a OS Config patch job.

  Args:
    parser: The parser for the command.
    verb: str, The verb to describe the resource, such as 'to describe'.
    plural: bool, If True, use a resource argument that returns a list.
  """
    concept_parsers.ConceptParser([CreatePatchJobResourceArg(verb, plural)]).AddToParser(parser)