from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerControllerDebuggeesBreakpointsListRequest(_messages.Message):
    """A ClouddebuggerControllerDebuggeesBreakpointsListRequest object.

  Fields:
    agentId: Identifies the agent. This is the ID returned in the
      RegisterDebuggee response.
    debuggeeId: Required. Identifies the debuggee.
    successOnTimeout: If set to `true` (recommended), returns
      `google.rpc.Code.OK` status and sets the `wait_expired` response field
      to `true` when the server-selected timeout has expired. If set to
      `false` (deprecated), returns `google.rpc.Code.ABORTED` status when the
      server-selected timeout has expired.
    waitToken: A token that, if specified, blocks the method call until the
      list of active breakpoints has changed, or a server-selected timeout has
      expired. The value should be set from the `next_wait_token` field in the
      last response. The initial value should be set to `"init"`.
  """
    agentId = _messages.StringField(1)
    debuggeeId = _messages.StringField(2, required=True)
    successOnTimeout = _messages.BooleanField(3)
    waitToken = _messages.StringField(4)