from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsExecutionValueValuesEnum(_messages.Enum):
    """Defines how Windows actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      WINDOWS_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      WINDOWS_EXECUTION_FORBIDDEN: Windows actions and worker pools are
        forbidden.
      WINDOWS_EXECUTION_UNRESTRICTED: No restrictions on execution of Windows
        actions.
      WINDOWS_EXECUTION_TERMINAL: Windows actions will always result in the
        worker VM being terminated after the action completes.
    """
    WINDOWS_EXECUTION_UNSPECIFIED = 0
    WINDOWS_EXECUTION_FORBIDDEN = 1
    WINDOWS_EXECUTION_UNRESTRICTED = 2
    WINDOWS_EXECUTION_TERMINAL = 3