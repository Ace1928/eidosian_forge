from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.network_security import API_VERSION_FOR_TRACK
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAddressGroupToParser(parser, release_track, resource_path):
    """Add project or organization address group argument."""
    address_group_data = yaml_data.ResourceYAMLData.FromPath(resource_path)
    resource_spec = concepts.ResourceSpec.FromYaml(address_group_data.GetData(), api_version=API_VERSION_FOR_TRACK[release_track])
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='ADDRESS_GROUP', concept_spec=resource_spec, required=True, group_help='address group group help.')
    concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)