from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import metrics
import six
def ConvertYamlToCamelCase(yaml_dict):
    """Recursively goes through the dictionary obj and replaces keys with the convert function.

  taken from:
  https://stackoverflow.com/questions/11700705/how-to-recursively-replace-character-in-keys-of-a-nested-dictionary.

  Args:
    yaml_dict: dict of loaded yaml

  Returns:
    A converted dict with camelCase keys
  """
    if isinstance(yaml_dict, (str, int, float)):
        return yaml_dict
    if isinstance(yaml_dict, dict):
        new = yaml_dict.__class__()
        for k, v in yaml_dict.items():
            new[SnakeToCamelCase(k)] = ConvertYamlToCamelCase(v)
    elif isinstance(yaml_dict, (list, set, tuple)):
        new = yaml_dict.__class__((ConvertYamlToCamelCase(v) for v in yaml_dict))
    else:
        return yaml_dict
    return new