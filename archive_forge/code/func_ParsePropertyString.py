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
def ParsePropertyString(property_string):
    """Parses a string into a section and property name.

  Args:
    property_string: str, The property string in the format section/property.

  Returns:
    (str, str), The section and property.  Both will be none if the input
    string is empty.  Property can be None if the string ends with a slash.
  """
    if not property_string:
        return (None, None)
    if '/' in property_string:
        section, prop = tuple(property_string.split('/', 1))
    else:
        section = None
        prop = property_string
    section = section or VALUES.default_section.name
    prop = prop or None
    return (section, prop)