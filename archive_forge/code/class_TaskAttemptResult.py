from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskAttemptResult(_messages.Message):
    """Result of a task attempt.

  Fields:
    exitCode: Optional. The exit code of this attempt. This may be unset if
      the container was unable to exit cleanly with a code due to some other
      failure. See status field for possible failure details.
    status: Optional. The status of this attempt. If the status code is OK,
      then the attempt succeeded.
  """
    exitCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    status = _messages.MessageField('GoogleRpcStatus', 2)