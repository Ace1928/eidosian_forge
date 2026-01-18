from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeviceGrant(_messages.Message):
    """Device grant. It is an authorization provided by the Spectrum Access
  System to a device to transmit using specified operating parameters after a
  successful heartbeat by the device.

  Enums:
    ChannelTypeValueValuesEnum: Type of channel used.
    StateValueValuesEnum: State of the grant.

  Fields:
    channelType: Type of channel used.
    expireTime: The expiration time of the grant.
    frequencyRange: The transmission frequency range.
    grantId: Grant Id.
    lastHeartbeatTransmitExpireTime: The transmit expiration time of the last
      heartbeat.
    maxEirp: Maximum Equivalent Isotropically Radiated Power (EIRP) permitted
      by the grant. The maximum EIRP is in units of dBm/MHz. The value of
      `maxEirp` represents the average (RMS) EIRP that would be measured by
      the procedure defined in FCC part 96.41(e)(3).
    moveList: The DPA move lists on which this grant appears.
    state: State of the grant.
    suspensionReason: If the grant is suspended, the reason(s) for suspension.
  """

    class ChannelTypeValueValuesEnum(_messages.Enum):
        """Type of channel used.

    Values:
      CHANNEL_TYPE_UNSPECIFIED: <no description>
      CHANNEL_TYPE_GAA: <no description>
      CHANNEL_TYPE_PAL: <no description>
    """
        CHANNEL_TYPE_UNSPECIFIED = 0
        CHANNEL_TYPE_GAA = 1
        CHANNEL_TYPE_PAL = 2

    class StateValueValuesEnum(_messages.Enum):
        """State of the grant.

    Values:
      GRANT_STATE_UNSPECIFIED: <no description>
      GRANT_STATE_GRANTED: The grant has been granted but the device is not
        heartbeating on it.
      GRANT_STATE_TERMINATED: The grant has been terminated by the SAS.
      GRANT_STATE_SUSPENDED: The grant has been suspended by the SAS.
      GRANT_STATE_AUTHORIZED: The device is currently transmitting.
      GRANT_STATE_EXPIRED: The grant has expired.
    """
        GRANT_STATE_UNSPECIFIED = 0
        GRANT_STATE_GRANTED = 1
        GRANT_STATE_TERMINATED = 2
        GRANT_STATE_SUSPENDED = 3
        GRANT_STATE_AUTHORIZED = 4
        GRANT_STATE_EXPIRED = 5
    channelType = _messages.EnumField('ChannelTypeValueValuesEnum', 1)
    expireTime = _messages.StringField(2)
    frequencyRange = _messages.MessageField('SasPortalFrequencyRange', 3)
    grantId = _messages.StringField(4)
    lastHeartbeatTransmitExpireTime = _messages.StringField(5)
    maxEirp = _messages.FloatField(6)
    moveList = _messages.MessageField('SasPortalDpaMoveList', 7, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    suspensionReason = _messages.StringField(9, repeated=True)