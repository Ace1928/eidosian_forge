from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphicsStatsBucket(_messages.Message):
    """A GraphicsStatsBucket object.

  Fields:
    frameCount: Number of frames in the bucket.
    renderMillis: Lower bound of render time in milliseconds.
  """
    frameCount = _messages.IntegerField(1)
    renderMillis = _messages.IntegerField(2)