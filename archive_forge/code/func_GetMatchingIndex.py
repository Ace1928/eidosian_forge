from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetMatchingIndex(index, func):
    """Returns index converted to a case that satisfies func."""
    if func(index):
        return index
    if not isinstance(index, six.string_types):
        return None
    for convert in [ConvertToCamelCase, ConvertToSnakeCase]:
        name = convert(index)
        if func(name):
            return name
    return None