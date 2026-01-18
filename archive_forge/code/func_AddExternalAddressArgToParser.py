from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddExternalAddressArgToParser(parser):
    """Sets up an argument for the external address resource."""
    address_data = yaml_data.ResourceYAMLData.FromPath('vmware.external_address')
    resource_spec = concepts.ResourceSpec.FromYaml(address_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='external_address', concept_spec=resource_spec, required=True, group_help='external_address.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)