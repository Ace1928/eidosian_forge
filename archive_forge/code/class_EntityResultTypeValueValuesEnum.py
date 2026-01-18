from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityResultTypeValueValuesEnum(_messages.Enum):
    """The result type for every entity in `entity_results`.

    Values:
      RESULT_TYPE_UNSPECIFIED: Unspecified. This value is never used.
      FULL: The key and properties.
      PROJECTION: A projected subset of properties. The entity may have no
        key.
      KEY_ONLY: Only the key.
    """
    RESULT_TYPE_UNSPECIFIED = 0
    FULL = 1
    PROJECTION = 2
    KEY_ONLY = 3