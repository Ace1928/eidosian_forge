from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
def _GetPropertyWithoutDefault(prop, properties_file):
    """Gets the given property without using a default.

  If the property has a designated command line argument and args is provided,
  check args for the value first. If the corresponding environment variable is
  set, use that second. Next, return whatever is in the property file.  Finally,
  use the callbacks to find values.  Do not check the default value.

  Args:
    prop: properties.Property, The property to get.
    properties_file: properties_file.PropertiesFile, An already loaded
      properties files to use.

  Returns:
    PropertyValue, The value of the property, or None if it is not set.
  """
    property_value = _GetPropertyWithoutCallback(prop, properties_file)
    if property_value and property_value.value is not None:
        return property_value
    for callback in prop.callbacks:
        value = callback()
        if value is not None:
            return PropertyValue(Stringize(value), PropertyValue.PropertySource.CALLBACK)
    if prop.is_feature_flag and prop != VALUES.core.enable_feature_flags and FeatureFlagEnabled():
        return PropertyValue(GetValueFromFeatureFlag(prop), PropertyValue.PropertySource.FEATURE_FLAG)
    return None