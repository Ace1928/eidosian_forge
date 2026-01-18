from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GarbageCollectionModeValueValuesEnum(_messages.Enum):
    """Immutable. Blockchain garbage collection mode.

    Values:
      GARBAGE_COLLECTION_MODE_UNSPECIFIED: The garbage collection has not been
        specified.
      FULL: Configures Geth's garbage collection so that older data not needed
        for a full node is deleted. This is the default mode when creating a
        full node.
      ARCHIVE: Configures Geth's garbage collection so that old data is never
        deleted. This is the default mode when creating an archive node. This
        value can also be chosen when creating a full node in order to create
        a partial/recent archive node. See [Sync
        modes](https://geth.ethereum.org/docs/fundamentals/sync-modes) for
        more details.
    """
    GARBAGE_COLLECTION_MODE_UNSPECIFIED = 0
    FULL = 1
    ARCHIVE = 2