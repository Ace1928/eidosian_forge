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
class KeyType(enum.Enum):
    CMEK = 'CMEK'
    CSEK = 'CSEK'