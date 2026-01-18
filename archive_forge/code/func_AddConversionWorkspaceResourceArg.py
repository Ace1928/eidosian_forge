from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddConversionWorkspaceResourceArg(parser, verb, positional=True):
    """Add a resource argument for a database migration conversion workspace.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'conversion_workspace'
    else:
        name = '--conversion-workspace'
    concept_parsers.ConceptParser.ForResource(name, GetConversionWorkspaceResourceSpec(), 'The conversion workspace {}.'.format(verb), required=True).AddToParser(parser)