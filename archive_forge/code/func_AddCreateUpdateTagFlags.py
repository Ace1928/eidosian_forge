from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCreateUpdateTagFlags():
    """Hook for adding flags to tags create and update."""
    resource_spec = concepts.ResourceSpec.FromYaml(yaml_data.ResourceYAMLData.FromPath('data_catalog.tag_template').GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='--tag-template', concept_spec=resource_spec, prefixes=True, required=True, flag_name_overrides={'project': '--tag-template-project'}, group_help="Tag template. `--tag-template-location` defaults to the tag's location.\n`--tag-template-project` defaults to the tag's project.\n      ")
    tag_template_arg = concept_parsers.ConceptParser([presentation_spec], command_level_fallthroughs={'--tag-template.location': ['--location'], '--tag-template.project': ['--project']})
    return [tag_template_arg]