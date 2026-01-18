from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Apk(_messages.Message):
    """An Android package file to install.

  Fields:
    location: The path to an APK to be installed on the device before the test
      begins.
    packageName: The java package for the APK to be installed. Value is
      determined by examining the application's manifest.
  """
    location = _messages.MessageField('FileReference', 1)
    packageName = _messages.StringField(2)