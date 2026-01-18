from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DartNormalizeTypeValueValuesEnum(_messages.Enum):
    """Type of normalization algorithm for boosted tree models using dart
    booster.

    Values:
      DART_NORMALIZE_TYPE_UNSPECIFIED: Unspecified dart normalize type.
      TREE: New trees have the same weight of each of dropped trees.
      FOREST: New trees have the same weight of sum of dropped trees.
    """
    DART_NORMALIZE_TYPE_UNSPECIFIED = 0
    TREE = 1
    FOREST = 2