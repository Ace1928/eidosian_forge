from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TreeValueValuesEnum(_messages.Enum):
    """Required. The tree to fetch.

    Values:
      DB_TREE_TYPE_UNSPECIFIED: Unspecified tree type.
      SOURCE_TREE: The source database tree.
      DRAFT_TREE: The draft database tree.
      DESTINATION_TREE: The destination database tree.
    """
    DB_TREE_TYPE_UNSPECIFIED = 0
    SOURCE_TREE = 1
    DRAFT_TREE = 2
    DESTINATION_TREE = 3