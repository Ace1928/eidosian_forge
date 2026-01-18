from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeviceSession(_messages.Message):
    """Protobuf message describing the device message, used from several RPCs.

  Enums:
    StateValueValuesEnum: Output only. Current state of the DeviceSession.

  Fields:
    activeStartTime: Output only. The timestamp that the session first became
      ACTIVE.
    androidDevice: Required. The requested device
    createTime: Output only. The time that the Session was created.
    displayName: Output only. The title of the DeviceSession to be presented
      in the UI.
    expireTime: Optional. If the device is still in use at this time, any
      connections will be ended and the SessionState will transition from
      ACTIVE to FINISHED.
    inactivityTimeout: Output only. The interval of time that this device must
      be interacted with before it transitions from ACTIVE to
      TIMEOUT_INACTIVITY.
    name: Optional. Name of the DeviceSession, e.g.
      "projects/{project_id}/deviceSessions/{session_id}"
    state: Output only. Current state of the DeviceSession.
    stateHistories: Output only. The historical state transitions of the
      session_state message including the current session state.
    ttl: Optional. The amount of time that a device will be initially
      allocated for. This can eventually be extended with the
      UpdateDeviceSession RPC. Default: 15 minutes.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the DeviceSession.

    Values:
      SESSION_STATE_UNSPECIFIED: Default value. This value is unused.
      REQUESTED: Initial state of a session request. The session is being
        validated for correctness and a device is not yet requested.
      PENDING: The session has been validated and is in the queue for a
        device.
      ACTIVE: The session has been granted and the device is accepting
        connections.
      EXPIRED: The session duration exceeded the device's reservation time
        period and timed out automatically.
      FINISHED: The user is finished with the session and it was canceled by
        the user while the request was still getting allocated or after
        allocation and during device usage period.
      UNAVAILABLE: Unable to complete the session because the device was
        unavailable and it failed to allocate through the scheduler. For
        example, a device not in the catalog was requested or the request
        expired in the allocation queue.
      ERROR: Unable to complete the session for an internal reason, such as an
        infrastructure failure.
    """
        SESSION_STATE_UNSPECIFIED = 0
        REQUESTED = 1
        PENDING = 2
        ACTIVE = 3
        EXPIRED = 4
        FINISHED = 5
        UNAVAILABLE = 6
        ERROR = 7
    activeStartTime = _messages.StringField(1)
    androidDevice = _messages.MessageField('AndroidDevice', 2)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    expireTime = _messages.StringField(5)
    inactivityTimeout = _messages.StringField(6)
    name = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    stateHistories = _messages.MessageField('SessionStateEvent', 9, repeated=True)
    ttl = _messages.StringField(10)