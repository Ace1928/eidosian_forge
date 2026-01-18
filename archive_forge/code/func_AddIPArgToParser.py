from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddIPArgToParser(parser):
    ip_address_id = yaml_data.ResourceYAMLData.FromPath('vmware.sddc.ip_address')
    resource_spec = concepts.ResourceSpec.FromYaml(ip_address_id.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='name', concept_spec=resource_spec, required=True, group_help='ip_address.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)