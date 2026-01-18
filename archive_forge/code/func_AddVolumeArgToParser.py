from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddVolumeArgToParser(parser, positional=False, group_help_text=None):
    """Sets up an argument for the instance resource."""
    if positional:
        name = 'volume'
    else:
        name = '--volume'
    volume_data = yaml_data.ResourceYAMLData.FromPath('bms.volume')
    resource_spec = concepts.ResourceSpec.FromYaml(volume_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help=group_help_text or 'volume.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)