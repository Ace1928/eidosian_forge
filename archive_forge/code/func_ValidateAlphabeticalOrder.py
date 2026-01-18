from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
def ValidateAlphabeticalOrder(self):
    """Validates whether the properties in the config file are in alphabetical order.

    Returns:
      InvalidOrderError: If the properties in config file are not in
          alphabetical order.
    """
    properties_list = list(self.parsed_yaml.keys())
    if properties_list != sorted(properties_list):
        return InvalidOrderError(properties_list=properties_list)
    return None