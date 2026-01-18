from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeStateValueValuesEnum(_messages.Enum):
    """Output only. The runtime (instance) state of the NotebookRuntime.

    Values:
      RUNTIME_STATE_UNSPECIFIED: Unspecified runtime state.
      RUNNING: NotebookRuntime is in running state.
      BEING_STARTED: NotebookRuntime is in starting state.
      BEING_STOPPED: NotebookRuntime is in stopping state.
      STOPPED: NotebookRuntime is in stopped state.
      BEING_UPGRADED: NotebookRuntime is in upgrading state. It is in the
        middle of upgrading process.
      ERROR: NotebookRuntime was unable to start/stop properly.
      INVALID: NotebookRuntime is in invalid state. Cannot be recovered.
    """
    RUNTIME_STATE_UNSPECIFIED = 0
    RUNNING = 1
    BEING_STARTED = 2
    BEING_STOPPED = 3
    STOPPED = 4
    BEING_UPGRADED = 5
    ERROR = 6
    INVALID = 7