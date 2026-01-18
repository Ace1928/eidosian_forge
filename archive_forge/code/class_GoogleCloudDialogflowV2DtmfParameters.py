from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DtmfParameters(_messages.Message):
    """The message in the response that indicates the parameters of DTMF.

  Fields:
    acceptsDtmfInput: Indicates whether DTMF input can be handled in the next
      request.
  """
    acceptsDtmfInput = _messages.BooleanField(1)