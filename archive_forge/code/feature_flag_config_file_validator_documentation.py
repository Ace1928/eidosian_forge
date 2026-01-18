from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
Validates the values of each property in the config file.

    This method ensures that the possible values of each property satisfy the
    property's validator.

    Args:
      values_list: list, list of possible values of the property in the config
          file.
      section_property: str, name of the property.

    Returns:
      InvalidPropertyError: If the property is not an actual Cloud SDK property.
      InvalidValueError: If the values do not satisfy the property's validator.
    