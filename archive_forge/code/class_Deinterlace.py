from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Deinterlace(_messages.Message):
    """Deinterlace configuration for input video.

  Fields:
    bwdif: Specifies the Bob Weaver Deinterlacing Filter Configuration.
    yadif: Specifies the Yet Another Deinterlacing Filter Configuration.
  """
    bwdif = _messages.MessageField('BwdifConfig', 1)
    yadif = _messages.MessageField('YadifConfig', 2)