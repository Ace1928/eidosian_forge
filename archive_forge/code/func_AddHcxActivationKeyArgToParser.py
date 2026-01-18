from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddHcxActivationKeyArgToParser(parser):
    """Sets up an argument for the HCX activation key resource."""
    hcx_activation_key_data = yaml_data.ResourceYAMLData.FromPath('vmware.hcx_activation_key')
    resource_spec = concepts.ResourceSpec.FromYaml(hcx_activation_key_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='hcx_activation_key', concept_spec=resource_spec, required=True, group_help='hcxactivationkey.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)