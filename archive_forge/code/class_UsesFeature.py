from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UsesFeature(_messages.Message):
    """A tag within a manifest.
  https://developer.android.com/guide/topics/manifest/uses-feature-
  element.html

  Fields:
    isRequired: The android:required value
    name: The android:name value
  """
    isRequired = _messages.BooleanField(1)
    name = _messages.StringField(2)