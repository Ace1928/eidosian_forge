from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddManagementDnsZoneBindingArgToParser(parser):
    """Sets up an argument for the management DNS zone binding resource."""
    path = 'vmware.management_dns_zone_binding'
    address_data = yaml_data.ResourceYAMLData.FromPath(path)
    resource_spec = concepts.ResourceSpec.FromYaml(address_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='management_dns_zone_binding', concept_spec=resource_spec, required=True, group_help='management_dns_zone_binding.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)