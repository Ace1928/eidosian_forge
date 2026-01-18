from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
def _ParseFeatureFlagsConfig(feature_flags_config_yaml):
    """Converts feature flag config file into a dictionary of Property objects.

  Args:
   feature_flags_config_yaml: str, feature flag config.

  Returns:
   property_dict: A dictionary of Property objects.
  """
    try:
        yaml_dict = yaml.load(feature_flags_config_yaml)
    except yaml.YAMLParseError as e:
        logging.warning('Unable to parse config: %s', e)
        return {}
    property_dict = {}
    for prop in yaml_dict or {}:
        yaml_prop = yaml_dict[prop]
        property_dict[prop] = Property(yaml_prop)
    return property_dict