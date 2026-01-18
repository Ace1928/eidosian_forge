from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _GetMetaDataValue(items, name, deserialize=False):
    """Gets the metadata value for the item in items with key == name.

  A metadata object is a list of dicts of the form:
    [
      {'key': key-name-1, 'value': field-1-value-string},
      {'key': key-name-2, 'value': field-2-value-string},
      ...
    ]

  Examples:
    x.metadata[windows-keys].email
      Deserializes the 'windows-keys' metadata value and gets the email value.
    x.metadata[windows-keys]
      Gets the 'windows-key' metadata string value.
    x.metadata[windows-keys][]
      Gets the deserialized 'windows-key' metadata value.

  Args:
    items: The metadata items list.
    name: The metadata name (which must match one of the 'key' values).
    deserialize: If True then attempt to deserialize a compact JSON string.

  Returns:
    The metadata value for name or None if not found or if items is not a
    metadata dict list.
  """
    item = _GetMetaDict(items, 'key', name)
    if item is None:
        return None
    value = item.get('value', None)
    if deserialize:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            pass
    return value