from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsensusClientValueValuesEnum(_messages.Enum):
    """Immutable. The consensus client.

    Values:
      CONSENSUS_CLIENT_UNSPECIFIED: Consensus client has not been specified,
        but should be.
      LIGHTHOUSE: Consensus client implementation written in Rust, maintained
        by Sigma Prime. See [Lighthouse - Sigma
        Prime](https://lighthouse.sigmaprime.io/) for details.
      ERIGON_EMBEDDED_CONSENSUS_LAYER: Erigon's embedded consensus client
        embedded in the execution client. Note this option is not currently
        available when creating new blockchain nodes. See [Erigon on
        GitHub](https://github.com/ledgerwatch/erigon#embedded-consensus-
        layer) for details.
    """
    CONSENSUS_CLIENT_UNSPECIFIED = 0
    LIGHTHOUSE = 1
    ERIGON_EMBEDDED_CONSENSUS_LAYER = 2