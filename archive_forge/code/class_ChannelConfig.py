from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChannelConfig(_messages.Message):
    """Configuration for a release channel.

  Fields:
    defaultVersion: Output only. Default version for this release channel,
      e.g.: "1.4.0".
  """
    defaultVersion = _messages.StringField(1)