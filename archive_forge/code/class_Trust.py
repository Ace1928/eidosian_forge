from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Trust(_messages.Message):
    """Represents a relationship between two domains. This allows a controller
  in one domain to authenticate a user in another domain. If the trust is
  being changed, it will be placed into the UPDATING state, which indicates
  that the resource is being reconciled. At this point, Get will reflect an
  intermediate state.

  Enums:
    StateValueValuesEnum: Output only. The current state of the trust.
    TrustDirectionValueValuesEnum: Required. The trust direction, which
      decides if the current domain is trusted, trusting, or both.
    TrustTypeValueValuesEnum: Required. The type of trust represented by the
      trust resource.

  Fields:
    createTime: Output only. The time the instance was created.
    lastTrustHeartbeatTime: Output only. The last heartbeat time when the
      trust was known to be connected.
    selectiveAuthentication: Optional. The trust authentication type, which
      decides whether the trusted side has forest/domain wide access or
      selective access to an approved set of resources.
    state: Output only. The current state of the trust.
    stateDescription: Output only. Additional information about the current
      state of the trust, if available.
    targetDnsIpAddresses: Required. The target DNS server IP addresses which
      can resolve the remote domain involved in the trust.
    targetDomainName: Required. The fully qualified target domain name which
      will be in trust with the current domain.
    trustDirection: Required. The trust direction, which decides if the
      current domain is trusted, trusting, or both.
    trustHandshakeSecret: Required. The trust secret used for the handshake
      with the target domain. This will not be stored.
    trustType: Required. The type of trust represented by the trust resource.
    updateTime: Output only. The last update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the trust.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: The domain trust is being created.
      UPDATING: The domain trust is being updated.
      DELETING: The domain trust is being deleted.
      CONNECTED: The domain trust is connected.
      DISCONNECTED: The domain trust is disconnected.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        UPDATING = 2
        DELETING = 3
        CONNECTED = 4
        DISCONNECTED = 5

    class TrustDirectionValueValuesEnum(_messages.Enum):
        """Required. The trust direction, which decides if the current domain is
    trusted, trusting, or both.

    Values:
      TRUST_DIRECTION_UNSPECIFIED: Not set.
      INBOUND: The inbound direction represents the trusting side.
      OUTBOUND: The outboud direction represents the trusted side.
      BIDIRECTIONAL: The bidirectional direction represents the trusted /
        trusting side.
    """
        TRUST_DIRECTION_UNSPECIFIED = 0
        INBOUND = 1
        OUTBOUND = 2
        BIDIRECTIONAL = 3

    class TrustTypeValueValuesEnum(_messages.Enum):
        """Required. The type of trust represented by the trust resource.

    Values:
      TRUST_TYPE_UNSPECIFIED: Not set.
      FOREST: The forest trust.
      EXTERNAL: The external domain trust.
    """
        TRUST_TYPE_UNSPECIFIED = 0
        FOREST = 1
        EXTERNAL = 2
    createTime = _messages.StringField(1)
    lastTrustHeartbeatTime = _messages.StringField(2)
    selectiveAuthentication = _messages.BooleanField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stateDescription = _messages.StringField(5)
    targetDnsIpAddresses = _messages.StringField(6, repeated=True)
    targetDomainName = _messages.StringField(7)
    trustDirection = _messages.EnumField('TrustDirectionValueValuesEnum', 8)
    trustHandshakeSecret = _messages.StringField(9)
    trustType = _messages.EnumField('TrustTypeValueValuesEnum', 10)
    updateTime = _messages.StringField(11)