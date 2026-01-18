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
def AllPropertyValues(self, list_unset=False, include_hidden=False, properties_file=None, only_file_contents=False):
    """Gets all the properties and their values for this section.

    Args:
      list_unset: bool, If True, include unset properties in the result.
      include_hidden: bool, True to include hidden properties in the result. If
        a property has a value set but is hidden, it will be included regardless
        of this setting.
      properties_file: properties_file.PropertiesFile, the file to read settings
        from.  If None the active property file will be used.
      only_file_contents: bool, True if values should be taken only from the
        properties file, false if flags, env vars, etc. should be consulted too.
        Mostly useful for listing file contents.

    Returns:
      {str:PropertyValue}, The dict of {property:value} for this section.
    """
    properties_file = properties_file or named_configs.ActivePropertiesFile.Load()
    result = {}
    for prop in self:
        if prop.is_internal:
            continue
        if prop.is_hidden and (not include_hidden) and (_GetPropertyWithoutCallback(prop, properties_file) is None):
            continue
        if only_file_contents:
            property_value = PropertyValue(properties_file.Get(prop.section, prop.name), PropertyValue.PropertySource.PROPERTY_FILE)
        else:
            property_value = _GetPropertyWithoutDefault(prop, properties_file)
        if property_value is None or property_value.value is None:
            if not list_unset:
                continue
            if prop.is_hidden and (not include_hidden):
                continue
        result[prop.name] = property_value
    return result