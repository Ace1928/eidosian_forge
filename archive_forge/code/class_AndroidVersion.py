from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidVersion(_messages.Message):
    """A version of the Android OS.

  Fields:
    apiLevel: The API level for this Android version. Examples: 18, 19.
    codeName: The code name for this Android version. Examples: "JellyBean",
      "KitKat".
    distribution: Market share for this version.
    id: An opaque id for this Android version. Use this id to invoke the
      TestExecutionService.
    releaseDate: The date this Android version became available in the market.
    tags: Tags for this dimension. Examples: "default", "preview",
      "deprecated".
    versionString: A string representing this version of the Android OS.
      Examples: "4.3", "4.4".
  """
    apiLevel = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    codeName = _messages.StringField(2)
    distribution = _messages.MessageField('Distribution', 3)
    id = _messages.StringField(4)
    releaseDate = _messages.MessageField('Date', 5)
    tags = _messages.StringField(6, repeated=True)
    versionString = _messages.StringField(7)