from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def __AsString(self, value):
    """Convert a value to appropriate string.

    Returns:
      String version of value with all carriage returns and line feeds removed.
    """
    if issubclass(self.__attribute.expected_type, str):
        cast_value = TYPE_STR(value)
    else:
        cast_value = TYPE_UNICODE(value)
    cast_value = cast_value.replace('\n', '')
    cast_value = cast_value.replace('\r', '')
    return cast_value