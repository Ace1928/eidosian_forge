from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidAppInfo(_messages.Message):
    """Android app information.

  Fields:
    name: The name of the app. Optional
    packageName: The package name of the app. Required.
    versionCode: The internal version code of the app. Optional.
    versionName: The version name of the app. Optional.
  """
    name = _messages.StringField(1)
    packageName = _messages.StringField(2)
    versionCode = _messages.StringField(3)
    versionName = _messages.StringField(4)