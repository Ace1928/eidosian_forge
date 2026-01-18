from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _ConvertToStringValue(prop):
    """Converts the prop to a string if it exists.

  Args:
    prop (object_value): The value returned from _GetAdditionalProperty.

  Returns:
    A string value for the given prop, or the `not_found_value` if the prop does
    not exist.
  """
    return getattr(prop, 'string_value', prop)