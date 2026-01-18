from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MapTargetTypeValueValuesEnum(_messages.Enum):
    """Optional. Will indicate how to represent a parquet map if present.

    Values:
      MAP_TARGET_TYPE_UNSPECIFIED: In this mode, we fall back to the default.
        Currently (3/24) we represent the map as: struct map_field_name {
        repeated struct key_value { key value } }
      ARRAY_OF_STRUCT: In this mode, we omit parquet's key_value struct and
        represent the map as: repeated struct map_field_name { key value }
    """
    MAP_TARGET_TYPE_UNSPECIFIED = 0
    ARRAY_OF_STRUCT = 1