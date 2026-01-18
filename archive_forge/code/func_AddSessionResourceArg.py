from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddSessionResourceArg(parser, verb, positional=True):
    """Add a resource argument for a Cloud Spanner session.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the session ID is a positional rather
      than a flag.
  """
    name = 'session' if positional else '--session'
    concept_parsers.ConceptParser.ForResource(name, GetSessionResourceSpec(), 'The Cloud Spanner session {}.'.format(verb), required=True).AddToParser(parser)