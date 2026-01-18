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
def AddCreateChannelArg(parser):
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('channel', ChannelResourceSpec(), 'Channel to create.', required=True), presentation_specs.ResourcePresentationSpec('--provider', ProviderResourceSpec(), 'Provider to use for the channel.', flag_name_overrides={'location': ''})], command_level_fallthroughs={'--provider.location': ['channel.location']}).AddToParser(parser)