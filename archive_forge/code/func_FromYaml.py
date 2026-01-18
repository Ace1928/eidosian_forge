from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@classmethod
def FromYaml(cls, yaml_data, is_positional=None, api_version=None):
    """Constructs an instance of ResourceSpec from yaml data.

    Args:
      yaml_data: dict, the parsed data from a resources.yaml file under
        command_lib/.
      is_positional: bool, optional value that determines if anchor argument is
        a positional and reformats anchor attribute name accordingly.
      api_version: string, overrides the default version in the resource
        registry if provided.

    Returns:
      A ResourceSpec object.
    """
    from googlecloudsdk.command_lib.util.apis import registry
    collection = registry.GetAPICollection(yaml_data['collection'], api_version=api_version)
    attributes = ParseAttributesFromData(yaml_data.get('attributes'), collection.detailed_params)
    return cls(resource_collection=collection.full_name, resource_name=yaml_data['name'], api_version=collection.api_version, disable_auto_completers=yaml_data.get('disable_auto_completers', ResourceSpec.disable_auto_complete), plural_name=yaml_data.get('plural_name'), is_positional=is_positional, **{attribute.parameter_name: attribute for attribute in attributes})