from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPrivateConnectionDeleteResourceArg(parser, verb, positional=True):
    """Add a resource argument for a database migration private connection.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'private_connection'
    else:
        name = '--private-connection'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetPrivateConnectionResourceSpec(), 'The private connection {}.'.format(verb), required=True)]
    concept_parsers.ConceptParser(resource_specs).AddToParser(parser)