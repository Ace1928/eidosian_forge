from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEnvironmentResourceArg(parser, verb, positional=True, required=True, plural=False):
    """Add a resource argument for a Cloud Composer Environment.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command
    verb: str, the verb to describe the resource, for example, 'to update'.
    positional: boolean, if True, means that the resource is a positional rather
        than a flag.
    required: boolean, if True, the arg is required
    plural: boolean, if True, expects a list of resources
  """
    noun = 'environment' + ('s' if plural else '')
    name = _BuildArgName(noun, positional)
    concept_parsers.ConceptParser.ForResource(name, GetEnvironmentResourceSpec(), 'The {} {}.'.format(noun, verb), required=required, plural=plural).AddToParser(parser)