from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddProviderResourceArg(parser, group_help_text, required=False):
    """Adds a resource argument for an Eventarc provider."""
    concept_parsers.ConceptParser.ForResource('provider', ProviderResourceSpec(), group_help_text, required=required).AddToParser(parser)