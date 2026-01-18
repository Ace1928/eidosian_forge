from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Overlay(_messages.Message):
    """Overlay configuration.

  Fields:
    animations: List of animations. The list should be chronological, without
      any time overlap.
    image: Image overlay.
  """
    animations = _messages.MessageField('Animation', 1, repeated=True)
    image = _messages.MessageField('Image', 2)