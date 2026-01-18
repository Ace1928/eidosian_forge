from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def GetValueFromObjectCustomMetadata(obj_metadata, search_key, default_value=None):
    """Filters a specific element out of an object's custom metadata.

  Args:
    obj_metadata: (apitools_messages.Object) The metadata for an object.
    search_key: (str) The custom metadata key to search for.
    default_value: (Any) The default value to use for the key if it cannot be
        found.

  Returns:
    (Tuple(bool, Any)) A tuple indicating if the value could be found in
    metadata and a value corresponding to search_key (the value at the specified
    key in custom metadata, or the default value if the specified key does not
    exist in the custom metadata).
  """
    try:
        value = next((attr.value for attr in obj_metadata.metadata.additionalProperties if attr.key == search_key), None)
        if value is None:
            return (False, default_value)
        return (True, value)
    except AttributeError:
        return (False, default_value)