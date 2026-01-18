from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
def ValidateValues(self, values_list, section_property):
    """Validates the values of each property in the config file.

    This method ensures that the possible values of each property satisfy the
    property's validator.

    Args:
      values_list: list, list of possible values of the property in the config
          file.
      section_property: str, name of the property.

    Returns:
      InvalidPropertyError: If the property is not an actual Cloud SDK property.
      InvalidValueError: If the values do not satisfy the property's validator.
    """
    try:
        section_name, property_name = section_property.split('/')
    except ValueError:
        return None
    try:
        section_instance = getattr(properties.VALUES, section_name)
    except AttributeError:
        return InvalidPropertyError(section_property, 'Property section [{}] does not exist.'.format(section_name))
    try:
        property_instance = getattr(section_instance, property_name)
    except AttributeError:
        return InvalidPropertyError(section_property, 'Property [{}] is not a property in section [{}].'.format(property_name, section_name))
    list_of_invalid_values = []
    for value in values_list:
        try:
            property_instance.Validate(value)
        except properties.InvalidValueError:
            list_of_invalid_values.append(value)
    if list_of_invalid_values:
        return InvalidValueError(invalid_values=list_of_invalid_values)
    return None