from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncounteredNonAndroidUiWidgetScreen(_messages.Message):
    """Additional details about encountered screens with elements that are not
  Android UI widgets.

  Fields:
    distinctScreens: Number of encountered distinct screens with non Android
      UI widgets.
    screenIds: Subset of screens which contain non Android UI widgets.
  """
    distinctScreens = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    screenIds = _messages.StringField(2, repeated=True)