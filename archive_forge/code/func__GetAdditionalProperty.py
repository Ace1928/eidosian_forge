from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _GetAdditionalProperty(properties, property_key, not_found_value='Unknown'):
    """Gets the value for the given key in a list of properties.

  Looks through a list of properties and tries to find the value for the given
  key. If it's not found, not_found_value is returned.

  Args:
    properties: A dictionary of key string, value string pairs.
    property_key: The key string for which we want to get the value.
    not_found_value: The string value to return if the key is not found.

  Returns:
    A string containing the value for the given key, or `not_found_value` if
    the key is not found.
  """
    for prop in properties:
        if prop.key == property_key:
            if hasattr(prop, 'value'):
                return prop.value
            break
    return not_found_value