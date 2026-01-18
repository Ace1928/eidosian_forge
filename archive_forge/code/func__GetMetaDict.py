from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _GetMetaDict(items, key, value):
    """Gets the dict in items that contains key==value.

  A metadict object is a list of dicts of the form:
    [
      {key: value-1, ...},
      {key: value-2, ...},
      ...
    ]

  Args:
    items: A list of dicts.
    key: The dict key name.
    value: The dict key value.

  Returns:
    The dict in items that contains key==value or None if no match or not a
    metadict.
  """
    try:
        for item in items:
            if item.get(key) == value:
                return item
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None