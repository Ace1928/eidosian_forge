from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def _ParseBoolean(value):
    """This is upstream logic from dev_appserver for parsing boolean arguments.

  Args:
    value: value assigned to a flag.

  Returns:
    A boolean parsed from value.

  Raises:
    ValueError: value.lower() is not in _TRUE_VALUES + _FALSE_VALUES.
  """
    if isinstance(value, bool):
        return value
    if value:
        value = value.lower()
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
        repr_value = (repr(value) for value in _TRUE_VALUES + _FALSE_VALUES)
        raise ValueError('%r unrecognized boolean; known booleans are %s.' % (value, ', '.join(repr_value)))
    return True