from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaDownload(_messages.Message):
    """Do not use this. For media support, add instead
  [][google.bytestream.RestByteStream] as an API to your configuration.

  Fields:
    enabled: Whether download is enabled.
  """
    enabled = _messages.BooleanField(1)