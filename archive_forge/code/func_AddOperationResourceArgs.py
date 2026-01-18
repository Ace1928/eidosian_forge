from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddOperationResourceArgs(parser, verb):
    concept_parsers.ConceptParser.ForResource('operation', GetOperationResourceSpec(), 'The name of the operation to {}.'.format(verb), required=True).AddToParser(parser)