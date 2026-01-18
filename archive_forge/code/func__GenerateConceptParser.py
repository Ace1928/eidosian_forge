from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def _GenerateConceptParser(self, resource_spec, attribute_names, repeated=False, shared_resource_flags=None, anchor_arg_name=None, group_help=None, is_required=False):
    """Generates a ConceptParser from YAMLConceptArgument.

    Args:
      resource_spec: concepts.ResourceSpec, used to create PresentationSpec
      attribute_names: names of resource attributes
      repeated: bool, whether or not the resource arg should be plural
      shared_resource_flags: [string], list of flags being generated elsewhere
      anchor_arg_name: string | None, anchor arg name
      group_help: string | None, group help text
      is_required: bool, whether the resource arg should be required

    Returns:
      ConceptParser that will be added to the parser.
    """
    shared_resource_flags = shared_resource_flags or []
    ignored_fields = list(concepts.IGNORED_FIELDS.values()) + self.removed_flags + shared_resource_flags
    no_gen = {n: '' for n in ignored_fields if n in attribute_names}
    command_level_fallthroughs = {}
    arg_fallthroughs = self.command_level_fallthroughs.copy()
    arg_fallthroughs.update({n: ['--' + n] for n in shared_resource_flags if n in attribute_names})
    concept_parsers.UpdateFallthroughsMap(command_level_fallthroughs, anchor_arg_name, arg_fallthroughs)
    presentation_spec_class = presentation_specs.ResourcePresentationSpec
    if isinstance(resource_spec, multitype.MultitypeResourceSpec):
        presentation_spec_class = presentation_specs.MultitypeResourcePresentationSpec
    return concept_parsers.ConceptParser([presentation_spec_class(anchor_arg_name, resource_spec, group_help=group_help, prefixes=False, required=is_required, flag_name_overrides=no_gen, plural=repeated)], command_level_fallthroughs=command_level_fallthroughs)