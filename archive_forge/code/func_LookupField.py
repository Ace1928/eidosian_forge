from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def LookupField(resource_key, fields):
    """Returns the actual_key match of resource_key in fields.

  Handles camelCase/snake_case key name variants for OnePlatform compatibility.
  Indices and slices in resource_key are ignored to normalize the lookup. This
  means that the lookup can determine the existence of an attribute name, but
  not a specific value among all repeated values.

  Args:
    resource_key: Ordered list of key names/indices, applied left to right. Each
      element in the list may be one of:
        str - A resource property name. This could be a class attribute name or
          a dict index.
        int - A list index. Selects one member is the list. Negative indices
          count from the end of the list, starting with -1 for the last element
          in the list. An out of bounds index is not an error; it produces the
          value None.
        None - A list slice. Selects all members of a list or dict like object.
          A slice of an empty dict or list is an empty dict or list.
    fields: The set of dotted field names to match against.

  Returns:
    The actual_key match of resource_key in fields or None if no match.
  """
    for convert in (lambda x: x, ConvertToCamelCase, ConvertToSnakeCase):
        actual_key = [convert(name) if isinstance(name, six.string_types) else name for name in resource_key]
        lookup_key = '.'.join([name for name in actual_key if isinstance(name, six.string_types)])
        if lookup_key in fields:
            return actual_key
    return None