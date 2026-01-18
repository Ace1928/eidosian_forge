from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DocumentReloadStatus(_messages.Message):
    """The status of a reload attempt.

  Fields:
    status: The status of a reload attempt or the initial load.
    time: The time of a reload attempt. This reload may have been triggered
      automatically or manually and may not have succeeded.
  """
    status = _messages.MessageField('GoogleRpcStatus', 1)
    time = _messages.StringField(2)