from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyUnitValueValuesEnum(_messages.Enum):
    """The unit for the key: e.g. 'key' or 'chunk'.

    Values:
      KEY_UNIT_UNSPECIFIED: Required default value
      KEY: Each entry corresponds to one key
      CHUNK: Each entry corresponds to a chunk of keys
    """
    KEY_UNIT_UNSPECIFIED = 0
    KEY = 1
    CHUNK = 2