from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosVersion(_messages.Message):
    """An iOS version.

  Fields:
    id: An opaque id for this iOS version. Use this id to invoke the
      TestExecutionService.
    majorVersion: An integer representing the major iOS version. Examples:
      "8", "9".
    minorVersion: An integer representing the minor iOS version. Examples:
      "1", "2".
    supportedXcodeVersionIds: The available Xcode versions for this version.
    tags: Tags for this dimension. Examples: "default", "preview",
      "deprecated".
  """
    id = _messages.StringField(1)
    majorVersion = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    minorVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    supportedXcodeVersionIds = _messages.StringField(4, repeated=True)
    tags = _messages.StringField(5, repeated=True)