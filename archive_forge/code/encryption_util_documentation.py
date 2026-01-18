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
Returns an EncryptionKey, None, or a CLEAR string constant.

  Args:
    sha256_hash (str): Attempts to return a CSEK key that matches this hash.
      Used for encrypting with a non-default key.
    url_for_missing_key_error (StorageUrl): If a key matching sha256_hash can
      not be found, raise an error adding this object URL to the error text.

  Returns:
    EncryptionKey: Custom or default key depending on presence of sha256_hash.
    None: Matching key to sha256_hash could not be found and
      url_for_missing_key_error was None. Or, no sha256_hash and no default key.
    user_request_args_factory.CLEAR (str): Value indicating that the
      user requested to clear an existing encryption.
  