from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ActionValueValueValuesEnum(_messages.Enum):
    """Only breakpoints with the specified action will pass the filter.

    Values:
      CAPTURE: Capture stack frame and variables and update the breakpoint.
        The data is only captured once. After that the breakpoint is set in a
        final state.
      LOG: Log each breakpoint hit. The breakpoint remains active until
        deleted or expired.
    """
    CAPTURE = 0
    LOG = 1