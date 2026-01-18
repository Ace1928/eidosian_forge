from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainTypeValueValuesEnum(_messages.Enum):
    """Immutable. The blockchain type of the node.

    Values:
      BLOCKCHAIN_TYPE_UNSPECIFIED: Blockchain type has not been specified, but
        should be.
      ETHEREUM: The blockchain type is Ethereum.
    """
    BLOCKCHAIN_TYPE_UNSPECIFIED = 0
    ETHEREUM = 1