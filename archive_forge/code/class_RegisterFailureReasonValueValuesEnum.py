from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegisterFailureReasonValueValuesEnum(_messages.Enum):
    """Output only. The reason the domain registration failed. Only set for
    domains in REGISTRATION_FAILED state.

    Values:
      REGISTER_FAILURE_REASON_UNSPECIFIED: Register failure unspecified.
      REGISTER_FAILURE_REASON_UNKNOWN: Registration failed for an unknown
        reason.
      DOMAIN_NOT_AVAILABLE: The domain is not available for registration.
      INVALID_CONTACTS: The provided contact information was rejected.
    """
    REGISTER_FAILURE_REASON_UNSPECIFIED = 0
    REGISTER_FAILURE_REASON_UNKNOWN = 1
    DOMAIN_NOT_AVAILABLE = 2
    INVALID_CONTACTS = 3