from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferResults(_messages.Message):
    """A message used in OfflineImportFeature and OnlineImportFeature to
  display the results of a transfer.

  Fields:
    bytesCopiedCount: Output only. The total number of bytes successfully
      copied to the destination.
    bytesFoundCount: Output only. The total number of bytes found.
    directoriesFoundCount: Output only. The number of directories found while
      listing. For example, if the root directory of the transfer is `base/`
      and there are two other directories, `a/` and `b/` under this directory,
      the count after listing `base/`, `base/a/` and `base/b/` is 3.
    endTime: Output only. The time that this transfer finished.
    errorLog: Output only. A URI to a file containing information about any
      files/directories that could not be transferred, or blank if there were
      no errors.
    objectsCopiedCount: Output only. The number of objects successfully copied
      to the destination.
    objectsFoundCount: Output only. The number of objects found.
  """
    bytesCopiedCount = _messages.IntegerField(1)
    bytesFoundCount = _messages.IntegerField(2)
    directoriesFoundCount = _messages.IntegerField(3)
    endTime = _messages.StringField(4)
    errorLog = _messages.StringField(5)
    objectsCopiedCount = _messages.IntegerField(6)
    objectsFoundCount = _messages.IntegerField(7)