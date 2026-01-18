from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnosticValueValuesEnum(_messages.Enum):
    """The diagnostic code specifies the local system's reason for the last
    change in session state. This allows remote systems to determine the
    reason that the previous session failed, for example. These diagnostic
    codes are specified in section 4.1 of RFC5880

    Values:
      ADMINISTRATIVELY_DOWN: <no description>
      CONCATENATED_PATH_DOWN: <no description>
      CONTROL_DETECTION_TIME_EXPIRED: <no description>
      DIAGNOSTIC_UNSPECIFIED: <no description>
      ECHO_FUNCTION_FAILED: <no description>
      FORWARDING_PLANE_RESET: <no description>
      NEIGHBOR_SIGNALED_SESSION_DOWN: <no description>
      NO_DIAGNOSTIC: <no description>
      PATH_DOWN: <no description>
      REVERSE_CONCATENATED_PATH_DOWN: <no description>
    """
    ADMINISTRATIVELY_DOWN = 0
    CONCATENATED_PATH_DOWN = 1
    CONTROL_DETECTION_TIME_EXPIRED = 2
    DIAGNOSTIC_UNSPECIFIED = 3
    ECHO_FUNCTION_FAILED = 4
    FORWARDING_PLANE_RESET = 5
    NEIGHBOR_SIGNALED_SESSION_DOWN = 6
    NO_DIAGNOSTIC = 7
    PATH_DOWN = 8
    REVERSE_CONCATENATED_PATH_DOWN = 9