from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DrmSystems(_messages.Message):
    """Defines configuration for DRM systems in use.

  Fields:
    clearkey: Clearkey configuration.
    fairplay: Fairplay configuration.
    playready: Playready configuration.
    widevine: Widevine configuration.
  """
    clearkey = _messages.MessageField('Clearkey', 1)
    fairplay = _messages.MessageField('Fairplay', 2)
    playready = _messages.MessageField('Playready', 3)
    widevine = _messages.MessageField('Widevine', 4)