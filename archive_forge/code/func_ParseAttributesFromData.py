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
def ParseAttributesFromData(attributes_data, expected_param_names):
    """Parses a list of ResourceParameterAttributeConfig from yaml data.

  Args:
    attributes_data: dict, the attributes data defined in
      command_lib/resources.yaml file.
    expected_param_names: [str], the names of the API parameters that the API
      method accepts. Example, ['projectsId', 'instancesId'].

  Returns:
    [ResourceParameterAttributeConfig].

  Raises:
    InvalidResourceArgumentLists: if the attributes defined in the yaml file
      don't match the expected fields in the API method.
  """
    raw_attributes = [ResourceParameterAttributeConfig.FromData(a) for a in attributes_data]
    registered_param_names = [a.parameter_name for a in raw_attributes]
    final_attributes = []
    for expected_name in expected_param_names:
        if raw_attributes and expected_name == raw_attributes[0].parameter_name:
            final_attributes.append(raw_attributes.pop(0))
        elif expected_name in IGNORED_FIELDS:
            attribute_name = IGNORED_FIELDS[expected_name]
            ignored_attribute = DEFAULT_RESOURCE_ATTRIBUTE_CONFIGS.get(attribute_name)
            ignored_attribute.parameter_name = expected_name
            final_attributes.append(ignored_attribute)
        else:
            raise InvalidResourceArgumentLists(expected_param_names, registered_param_names)
    if raw_attributes:
        raise InvalidResourceArgumentLists(expected_param_names, registered_param_names)
    return final_attributes