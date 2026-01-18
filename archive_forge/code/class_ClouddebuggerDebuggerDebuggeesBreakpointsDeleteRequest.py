from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerDebuggerDebuggeesBreakpointsDeleteRequest(_messages.Message):
    """A ClouddebuggerDebuggerDebuggeesBreakpointsDeleteRequest object.

  Fields:
    breakpointId: Required. ID of the breakpoint to delete.
    clientVersion: Required. The client version making the call. Schema:
      `domain/type/version` (e.g., `google.com/intellij/v1`).
    debuggeeId: Required. ID of the debuggee whose breakpoint to delete.
  """
    breakpointId = _messages.StringField(1, required=True)
    clientVersion = _messages.StringField(2)
    debuggeeId = _messages.StringField(3, required=True)