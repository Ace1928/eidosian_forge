from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddNamespaceResourceArg(parser, verb, positional=True):
    """Adds a resource argument for a Service Directory namespace."""
    name = 'namespace' if positional else '--namespace'
    return concept_parsers.ConceptParser.ForResource(name, GetNamespaceResourceSpec(), 'The Service Directory namespace {}'.format(verb), required=True).AddToParser(parser)