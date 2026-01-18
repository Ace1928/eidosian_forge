from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPrivateConnectionToParser(parser, positional=False):
    """Sets up an argument for the Private Connection resource."""
    name = '--private-connection'
    if positional:
        name = 'private_connection'
    private_connection_data = yaml_data.ResourceYAMLData.FromPath('vmware.private_connection')
    resource_spec = concepts.ResourceSpec.FromYaml(private_connection_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='private_connection.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)