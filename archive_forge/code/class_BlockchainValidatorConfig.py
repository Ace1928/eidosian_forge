from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainValidatorConfig(_messages.Message):
    """Represents the configuration of a blockchain validator, as it would be
  configured on a validator client.

  Enums:
    BlockchainTypeValueValuesEnum: Immutable. The blockchain type of the
      validator.
    KeySourceValueValuesEnum: Immutable. The source of the voting key for the
      blockchain validator.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    blockchainNodeId: Optional. The fully qualified name of the blockchain
      node which carries out work on behalf of the validator. If not set, the
      validator must either be operated outside of Blockchain Validator
      Manager, or it will be offline (no attestations or blocks will be
      produced). If this node is offline or deleted, the validator will be
      offline.
    blockchainType: Immutable. The blockchain type of the validator.
    createTime: Output only. [Output only] Create time stamp
    ethereumProtocolDetails: Optional. Ethereum-specific configuration for a
      blockchain validator.
    existingSeedPhraseReference: Optional. An existing seed phrase, read from
      Secret Manager.
    keySource: Immutable. The source of the voting key for the blockchain
      validator.
    labels: Optional. Labels as key value pairs
    name: Identifier. The name of the validator. It must have the format `"pro
      jects/{project}/locations/{location}/blockchainValidatorConfigs/{validat
      or}"`. `{validator}` must contain only letters (`[A-Za-z]`), numbers
      (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`),
      plus (`+`) or percent signs (`%`). It must be between 3 and 255
      characters in length, and it must not start with `"goog"`.
    remoteWeb3Signer: Optional. Connection details of a remote Web3Signer
      service to use for signing attestations and blocks.
    seedPhraseReference: Optional. A new seed phrase, optionally written to
      Secret Manager.
    updateTime: Output only. [Output only] Update time stamp
    validationWorkEnabled: Required. True if the blockchain node requests and
      signs attestations and blocks on behalf of this validator, false if not.
      This does NOT define whether the blockchain expects work to occur, only
      whether the blockchain node specified above is carrying out validation
      tasks. This should be enabled under normal conditions, but may be useful
      when migrating validators to/from Blockchain Node Engine, where the
      validator may be paused during the migration.
    votingPublicKey: Output only. Immutable. The public key identifier of the
      validator, as a hexadecimal string prefixed with "0x". Note content of
      this field varies depending on the blockchain. This is provided by the
      server when creating or importing keys, and copied from the remote key
      signer configuration when configuring an external signing service.
  """

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    blockchainNodeId = _messages.StringField(1)
    blockchainType = _messages.EnumField('BlockchainTypeValueValuesEnum', 2)
    createTime = _messages.StringField(3)
    ethereumProtocolDetails = _messages.MessageField('EthereumDetails', 4)
    existingSeedPhraseReference = _messages.MessageField('ExistingSeedPhraseReference', 5)
    keySource = _messages.EnumField('KeySourceValueValuesEnum', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    remoteWeb3Signer = _messages.MessageField('RemoteWeb3Signer', 9)
    seedPhraseReference = _messages.MessageField('SeedPhraseReference', 10)
    updateTime = _messages.StringField(11)
    validationWorkEnabled = _messages.BooleanField(12)
    votingPublicKey = _messages.StringField(13)