from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RegularFile(_messages.Message):
    """A file or directory to install on the device before the test starts.

  Fields:
    content: Required. The source file.
    devicePath: Required. Where to put the content on the device. Must be an
      absolute, allowlisted path. If the file exists, it will be replaced. The
      following device-side directories and any of their subdirectories are
      allowlisted: ${EXTERNAL_STORAGE}, /sdcard, or /storage
      ${ANDROID_DATA}/local/tmp, or /data/local/tmp Specifying a path outside
      of these directory trees is invalid. The paths /sdcard and /data will be
      made available and treated as implicit path substitutions. E.g. if
      /sdcard on a particular device does not map to external storage, the
      system will replace it with the external storage path prefix for that
      device and copy the file there. It is strongly advised to use the
      Environment API in app and test code to access files on the device in a
      portable way.
  """
    content = _messages.MessageField('FileReference', 1)
    devicePath = _messages.StringField(2)