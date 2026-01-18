from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerDebuggerDebuggeesListRequest(_messages.Message):
    """A ClouddebuggerDebuggerDebuggeesListRequest object.

  Fields:
    clientVersion: Required. The client version making the call. Schema:
      `domain/type/version` (e.g., `google.com/intellij/v1`).
    includeInactive: When set to `true`, the result includes all debuggees.
      Otherwise, the result includes only debuggees that are active.
    project: Required. Project number of a Google Cloud project whose
      debuggees to list.
  """
    clientVersion = _messages.StringField(1)
    includeInactive = _messages.BooleanField(2)
    project = _messages.StringField(3)