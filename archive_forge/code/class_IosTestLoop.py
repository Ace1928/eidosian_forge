from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IosTestLoop(_messages.Message):
    """A game loop test of an iOS application.

  Fields:
    bundleId: Bundle ID of the app.
  """
    bundleId = _messages.StringField(1)