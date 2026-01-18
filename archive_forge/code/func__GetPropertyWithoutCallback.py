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
def _GetPropertyWithoutCallback(prop, properties_file):
    """Gets the given property without using a callback or default.

  If the property has a designated command line argument and args is provided,
  check args for the value first. If the corresponding environment variable is
  set, use that second. Finally, return whatever is in the property file.  Do
  not check for values in callbacks or defaults.

  Args:
    prop: properties.Property, The property to get.
    properties_file: PropertiesFile, An already loaded properties files to use.

  Returns:
    PropertyValue, The value of the property, or None if it is not set.
  """
    invocation_stack = VALUES.GetInvocationStack()
    for value_flags in reversed(invocation_stack):
        if prop not in value_flags:
            continue
        value_flag = value_flags.get(prop, None)
        if not value_flag:
            continue
        if value_flag.value is not None:
            return PropertyValue(Stringize(value_flag.value), PropertyValue.PropertySource.FLAG)
    value = encoding.GetEncodedValue(os.environ, prop.EnvironmentName())
    if value is not None:
        return PropertyValue(Stringize(value), PropertyValue.PropertySource.ENVIRONMENT)
    value = properties_file.Get(prop.section, prop.name)
    if value is not None:
        return PropertyValue(Stringize(value), PropertyValue.PropertySource.PROPERTY_FILE)
    return None