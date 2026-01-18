from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerDebuggerDebuggeesBreakpointsListRequest(_messages.Message):
    """A ClouddebuggerDebuggerDebuggeesBreakpointsListRequest object.

  Enums:
    ActionValueValueValuesEnum: Only breakpoints with the specified action
      will pass the filter.

  Fields:
    action_value: Only breakpoints with the specified action will pass the
      filter.
    clientVersion: Required. The client version making the call. Schema:
      `domain/type/version` (e.g., `google.com/intellij/v1`).
    debuggeeId: Required. ID of the debuggee whose breakpoints to list.
    includeAllUsers: When set to `true`, the response includes the list of
      breakpoints set by any user. Otherwise, it includes only breakpoints set
      by the caller.
    includeInactive: When set to `true`, the response includes active and
      inactive breakpoints. Otherwise, it includes only active breakpoints.
    stripResults: This field is deprecated. The following fields are always
      stripped out of the result: `stack_frames`, `evaluated_expressions` and
      `variable_table`.
    waitToken: A wait token that, if specified, blocks the call until the
      breakpoints list has changed, or a server selected timeout has expired.
      The value should be set from the last response. The error code
      `google.rpc.Code.ABORTED` (RPC) is returned on wait timeout, which
      should be called again with the same `wait_token`.
  """

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
    action_value = _messages.EnumField('ActionValueValueValuesEnum', 1)
    clientVersion = _messages.StringField(2)
    debuggeeId = _messages.StringField(3, required=True)
    includeAllUsers = _messages.BooleanField(4)
    includeInactive = _messages.BooleanField(5)
    stripResults = _messages.BooleanField(6)
    waitToken = _messages.StringField(7)