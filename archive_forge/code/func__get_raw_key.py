from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import collections
import enum
import hashlib
import re
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.cache import function_result_cache
def _get_raw_key(args, key_field_name):
    """Searches for key values in flags, falling back to a file if necessary.

  Args:
    args: An object containing flag values from the command surface.
    key_field_name (str): Corresponds to a flag name or field name in the key
        file.

  Returns:
    The flag value associated with key_field_name, or the value contained in the
    key file.
  """
    flag_key = getattr(args, key_field_name, None)
    if flag_key is not None:
        return flag_key
    return _read_key_store_file().get(key_field_name)