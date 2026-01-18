from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddRecognizerArgToParser(parser):
    """Sets up an argument for the recognizer resource."""
    resource_data = yaml_data.ResourceYAMLData.FromPath('ml.speech.recognizer')
    resource_spec = concepts.ResourceSpec.FromYaml(resource_data.GetData(), api_version='v2')
    presentation_spec = presentation_specs.ResourcePresentationSpec(name='recognizer', concept_spec=resource_spec, required=True, group_help='recognizer.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)