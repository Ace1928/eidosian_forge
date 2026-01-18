from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EthereumNodeDetails(_messages.Message):
    """Ethereum-specific blockchain node details.

  Enums:
    ConsensusClientValueValuesEnum: Required. The consensus client.
    ExecutionClientValueValuesEnum: Required. The execution client
    NetworkValueValuesEnum: Immutable. The Ethereum environment being
      accessed.

  Fields:
    consensusClient: Required. The consensus client.
    executionClient: Required. The execution client
    mevRelayUrls: Optional. URLs for MEV-relay services to use for block
      building. When set, a GCP-managed MEV-boost service is configured on the
      beacon client.
    network: Immutable. The Ethereum environment being accessed.
  """

    class ConsensusClientValueValuesEnum(_messages.Enum):
        """Required. The consensus client.

    Values:
      CONSENSUS_CLIENT_UNSPECIFIED: Consensus client has not been specified,
        but should be.
      LIGHTHOUSE: Consensus client implementation written in Rust, maintained
        by Sigma Prime. See [Lighthouse - Sigma
        Prime](https://lighthouse.sigmaprime.io/) for details.
    """
        CONSENSUS_CLIENT_UNSPECIFIED = 0
        LIGHTHOUSE = 1

    class ExecutionClientValueValuesEnum(_messages.Enum):
        """Required. The execution client

    Values:
      EXECUTION_CLIENT_UNSPECIFIED: Execution client has not been specified,
        but should be.
      GETH: Official Go implementation of the Ethereum protocol. See [go-
        ethereum](https://geth.ethereum.org/) for details.
    """
        EXECUTION_CLIENT_UNSPECIFIED = 0
        GETH = 1

    class NetworkValueValuesEnum(_messages.Enum):
        """Immutable. The Ethereum environment being accessed.

    Values:
      NETWORK_UNSPECIFIED: The network has not been specified, but should be.
      MAINNET: The Ethereum Mainnet.
      TESTNET_GOERLI_PRATER: Deprecated: The Ethereum Testnet based on Goerli
        protocol. Holesky (TESTNET_HOLESKY) is the recommended testnet to
        replace Goerli.
      TESTNET_SEPOLIA: The Ethereum Testnet based on Sepolia/Bepolia protocol.
        See https://github.com/eth-clients/sepolia.
      TESTNET_HOLESKY: The Ethereum Testnet based on Holesky specification.
        See https://github.com/eth-clients/holesky.
    """
        NETWORK_UNSPECIFIED = 0
        MAINNET = 1
        TESTNET_GOERLI_PRATER = 2
        TESTNET_SEPOLIA = 3
        TESTNET_HOLESKY = 4
    consensusClient = _messages.EnumField('ConsensusClientValueValuesEnum', 1)
    executionClient = _messages.EnumField('ExecutionClientValueValuesEnum', 2)
    mevRelayUrls = _messages.StringField(3, repeated=True)
    network = _messages.EnumField('NetworkValueValuesEnum', 4)