from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainValidatorConfigTemplate(_messages.Message):
    """A templatised set of blockchain validator configs, from which multiple
  configurations can be generated.

  Enums:
    BlockchainNodeSourceValueValuesEnum: Immutable. The source of the
      blockchain node for the validator configurations to be deployed to.
    BlockchainTypeValueValuesEnum: Immutable. The blockchain type of the
      validator.
    KeySourceValueValuesEnum: Immutable. The source of the voting key for the
      blockchain validator.

  Fields:
    blockchainNodeSource: Immutable. The source of the blockchain node for the
      validator configurations to be deployed to.
    blockchainType: Immutable. The blockchain type of the validator.
    ethereumProtocolDetails: Ethereum-specific configuration for a blockchain
      validator.
    existingBlockchainNodeSource: Configuration for deploying blockchain
      validators to an existing blockchain node.
    existingSeedPhraseReference: Optional. An existing seed phrase, read from
      Secret Manager.
    keySource: Immutable. The source of the voting key for the blockchain
      validator.
    newBlockchainNodeSource: Configuration for creating a new blockchain node
      to deploy the blockchain validator(s) to.
    remoteWeb3Signer: Optional. Connection details of a remote Web3Signer
      service to use for signing attestations and blocks.
    seedPhraseReference: Optional. A new seed phrase, optionally written to
      Secret Manager.
    validationWorkEnabled: Required. True if the blockchain node requests and
      signs attestations and blocks on behalf of this validator, false if not.
      This does NOT define whether the blockchain expects work to occur, only
      whether the blockchain node specified above is carrying out validation
      tasks.
  """

    class BlockchainNodeSourceValueValuesEnum(_messages.Enum):
        """Immutable. The source of the blockchain node for the validator
    configurations to be deployed to.

    Values:
      BLOCKCHAIN_NODE_SOURCE_UNSPECIFIED: Blockchain node source has not been
        specified, but should be.
      NEW_BLOCKCHAIN_NODE: Create a new blockchain node to deploy the
        validators to.
      EXISTING_BLOCKCHAIN_NODE: Deploying blockchain validators to an existing
        blockchain node, or to no node.
    """
        BLOCKCHAIN_NODE_SOURCE_UNSPECIFIED = 0
        NEW_BLOCKCHAIN_NODE = 1
        EXISTING_BLOCKCHAIN_NODE = 2

    class BlockchainTypeValueValuesEnum(_messages.Enum):
        """Immutable. The blockchain type of the validator.

    Values:
      BLOCKCHAIN_TYPE_UNSPECIFIED: Blockchain type has not been specified, but
        should be.
      ETHEREUM: The blockchain type is Ethereum.
    """
        BLOCKCHAIN_TYPE_UNSPECIFIED = 0
        ETHEREUM = 1

    class KeySourceValueValuesEnum(_messages.Enum):
        """Immutable. The source of the voting key for the blockchain validator.

    Values:
      KEY_SOURCE_UNSPECIFIED: Voting key source has not been specified, but
        should be.
      REMOTE_WEB3_SIGNER: The voting key is stored in a remote signing service
        (Web3Signer) and signing requests are delegated.
      SEED_PHRASE_REFERENCE: Derive voting keys from new seed material.
      EXISTING_SEED_PHRASE_REFERENCE: Derive voting keys from existing seed
        material.
    """
        KEY_SOURCE_UNSPECIFIED = 0
        REMOTE_WEB3_SIGNER = 1
        SEED_PHRASE_REFERENCE = 2
        EXISTING_SEED_PHRASE_REFERENCE = 3
    blockchainNodeSource = _messages.EnumField('BlockchainNodeSourceValueValuesEnum', 1)
    blockchainType = _messages.EnumField('BlockchainTypeValueValuesEnum', 2)
    ethereumProtocolDetails = _messages.MessageField('EthereumDetailsTemplate', 3)
    existingBlockchainNodeSource = _messages.MessageField('ExistingBlockchainNodeSource', 4)
    existingSeedPhraseReference = _messages.MessageField('ExistingSeedPhraseReferenceTemplate', 5)
    keySource = _messages.EnumField('KeySourceValueValuesEnum', 6)
    newBlockchainNodeSource = _messages.MessageField('NewBlockchainNodeSource', 7)
    remoteWeb3Signer = _messages.MessageField('RemoteWeb3SignerTemplate', 8)
    seedPhraseReference = _messages.MessageField('SeedPhraseReferenceTemplate', 9)
    validationWorkEnabled = _messages.BooleanField(10)