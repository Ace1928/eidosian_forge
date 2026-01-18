from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
def ValidateValueTypes(self, values_list):
    """Validates the values of each property in the config file.

    This method ensures that the values of each property are of the same type.

    Args:
      values_list: list, list of possible values of the property in the config
          file.

    Returns:
      InconsistentValuesError: If the values are not of the same type.
    """
    if not values_list:
        return None
    first_value_type = type(values_list[0])
    for value in values_list:
        if not isinstance(value, first_value_type):
            return InconsistentValuesError(values=values_list)
    return None