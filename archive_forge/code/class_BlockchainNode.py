from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainNode(_messages.Message):
    """A representation of a blockchain node.

  Enums:
    BlockchainTypeValueValuesEnum: Immutable. The blockchain type of the node.
    StateValueValuesEnum: Output only. A status representing the state of the
      node.

  Messages:
    LabelsValue: User-provided key-value pairs.

  Fields:
    blockchainType: Immutable. The blockchain type of the node.
    connectionInfo: Output only. The connection information used to interact
      with a blockchain node.
    createTime: Output only. The timestamp at which the blockchain node was
      first created.
    ethereumDetails: Ethereum-specific blockchain node details.
    labels: User-provided key-value pairs.
    name: Output only. The fully qualified name of the blockchain node. e.g.
      `projects/my-project/locations/us-central1/blockchainNodes/my-node`.
    privateServiceConnectEnabled: Optional. When true, the node is only
      accessible via Private Service Connect; no public endpoints are exposed.
      Otherwise, the node is only accessible via public endpoints. Warning:
      Private Service Connect enabled nodes may require a manual migration
      effort to remain compatible with future versions of the product. If this
      feature is enabled, you will be notified of these changes along with any
      required action to avoid disruption. See
      https://cloud.google.com/vpc/docs/private-service-connect.
    state: Output only. A status representing the state of the node.
    updateTime: Output only. The timestamp at which the blockchain node was
      last updated.
  """

    class BlockchainTypeValueValuesEnum(_messages.Enum):
        """Immutable. The blockchain type of the node.

    Values:
      BLOCKCHAIN_TYPE_UNSPECIFIED: Blockchain type has not been specified, but
        should be.
      ETHEREUM: The blockchain type is Ethereum.
    """
        BLOCKCHAIN_TYPE_UNSPECIFIED = 0
        ETHEREUM = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. A status representing the state of the node.

    Values:
      STATE_UNSPECIFIED: The state has not been specified.
      CREATING: The node has been requested and is in the process of being
        created.
      DELETING: The existing node is undergoing deletion, but is not yet
        finished.
      RUNNING: The node is running and ready for use.
      ERROR: The node is in an unexpected or errored state.
      UPDATING: The node is currently being updated.
      REPAIRING: The node is currently being repaired.
      RECONCILING: The node is currently being reconciled.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        DELETING = 2
        RUNNING = 3
        ERROR = 4
        UPDATING = 5
        REPAIRING = 6
        RECONCILING = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-provided key-value pairs.

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
    blockchainType = _messages.EnumField('BlockchainTypeValueValuesEnum', 1)
    connectionInfo = _messages.MessageField('ConnectionInfo', 2)
    createTime = _messages.StringField(3)
    ethereumDetails = _messages.MessageField('EthereumDetails', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    privateServiceConnectEnabled = _messages.BooleanField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)