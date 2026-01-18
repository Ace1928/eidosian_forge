from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSchemaResourceArg(parser, verb, plural=False):
    """Add a resource argument for a Cloud Pub/Sub Schema.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    plural: bool, if True, use a resource argument that returns a list.
  """
    concept_parsers.ConceptParser([CreateSchemaResourceArg(verb, plural=plural)]).AddToParser(parser)