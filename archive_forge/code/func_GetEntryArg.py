from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetEntryArg():
    entry_data = yaml_data.ResourceYAMLData.FromPath('data_catalog.entry')
    resource_spec = concepts.ResourceSpec.FromYaml(entry_data.GetData(), is_positional=True)
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='entry', concept_spec=resource_spec, group_help='Entry to update.')
    return concept_parsers.ConceptParser([presentation_spec])