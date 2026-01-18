from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetMessageFieldType(resource_key, message):
    """Returns the messages module type for key in message and the actual key.

  Handles camelCase/snake_case key name variants for OnePlatform compatibility.
  Indices and slices in resource_key are ignored -- they are not needed for
  repeated field queries.

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
    message: The known proto message type if not None.

  Raises:
    KeyError: If key is not in message.

  Returns:
    (type, actual_key), the messages module type for key in message and the
      actual key (names in the proper case, indices omitted).
  """
    actual_key = []
    for name in resource_key:
        if not isinstance(name, six.string_types):
            continue
        for convert in (lambda x: x, ConvertToCamelCase, ConvertToSnakeCase):
            actual_name = convert(name)
            try:
                message = message.field_by_name(actual_name).type
            except (AttributeError, KeyError):
                pass
            else:
                break
        else:
            raise KeyError('Field {} not in message.'.format(name))
        actual_key.append(actual_name)
    if message == six.integer_types:
        message = int
    return (message, actual_key)