from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegistrationStatus(_messages.Message):
    """RegistrationStatus describes the certificate provisioning status of a
  WorkloadRegistration resource.

  Enums:
    StateValueValuesEnum: The current state of registration.

  Fields:
    state: The current state of registration.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of registration.

    Values:
      REGISTRATION_STATE_UNSPECIFIED: REGISTRATION_STATE_UNSPECIFIED is the
        default value.
      REGISTRATION_STATE_READY: REGISTRATION_STATE_READY indicates that the
        registration is ready.
      REGISTRATION_STATE_IN_PROGRESS: REGISTRATION_STATE_IN_PROGRESS indicates
        that the registration is in progress.
      REGISTRATION_STATE_INTERNAL_ERROR: REGISTRATION_STATE_INTERNAL_ERROR
        indicates that the registration has encountered some internal errors
        but is retrying. Contact support if this persists.
    """
        REGISTRATION_STATE_UNSPECIFIED = 0
        REGISTRATION_STATE_READY = 1
        REGISTRATION_STATE_IN_PROGRESS = 2
        REGISTRATION_STATE_INTERNAL_ERROR = 3
    state = _messages.EnumField('StateValueValuesEnum', 1)