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
def AddOsImageArgToParser(parser, positional=False):
    """Sets up an argument for an os-image resource."""
    if positional:
        name = 'os_image'
    else:
        name = '--os-image'
    os_image_data = yaml_data.ResourceYAMLData.FromPath('bms.os_image')
    resource_spec = concepts.ResourceSpec.FromYaml(os_image_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='os_image.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)