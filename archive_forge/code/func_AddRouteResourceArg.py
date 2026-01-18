from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddRouteResourceArg(parser, verb, positional=True):
    """Add a resource argument for a Datastream route.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to create'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'route'
    else:
        name = '--route'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetRouteResourceSpec(), 'The route {}.'.format(verb), required=True)]
    concept_parsers.ConceptParser(resource_specs).AddToParser(parser)