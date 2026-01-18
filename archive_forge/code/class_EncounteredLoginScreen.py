from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncounteredLoginScreen(_messages.Message):
    """Additional details about encountered login screens.

  Fields:
    distinctScreens: Number of encountered distinct login screens.
    screenIds: Subset of login screens.
  """
    distinctScreens = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    screenIds = _messages.StringField(2, repeated=True)