from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
def GetJsonValues(self, value):
    """Convert the user input values to JSON value or JSON array value.

    Args:
      value: String or string list, the user input values of the column.

    Returns:
      extra_types.JsonArray or extra_types.JsonValue, the json value of a single
          column in the format that API accepts.
    """
    return self.col_type.GetJsonValue(value)