from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerDebuggerDebuggeesBreakpointsGetRequest(_messages.Message):
    """A ClouddebuggerDebuggerDebuggeesBreakpointsGetRequest object.

  Fields:
    breakpointId: Required. ID of the breakpoint to get.
    clientVersion: Required. The client version making the call. Schema:
      `domain/type/version` (e.g., `google.com/intellij/v1`).
    debuggeeId: Required. ID of the debuggee whose breakpoint to get.
  """
    breakpointId = _messages.StringField(1, required=True)
    clientVersion = _messages.StringField(2)
    debuggeeId = _messages.StringField(3, required=True)