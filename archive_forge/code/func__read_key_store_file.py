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
@function_result_cache.lru(maxsize=1)
def _read_key_store_file():
    key_store_path = properties.VALUES.storage.key_store_path.Get()
    if not key_store_path:
        return {}
    return yaml.load_path(key_store_path)