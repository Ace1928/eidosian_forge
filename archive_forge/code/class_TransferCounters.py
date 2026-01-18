from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferCounters(_messages.Message):
    """A collection of counters that report the progress of a transfer
  operation.

  Fields:
    bytesCopiedToSink: Bytes that are copied to the data sink.
    bytesDeletedFromSink: Bytes that are deleted from the data sink.
    bytesDeletedFromSource: Bytes that are deleted from the data source.
    bytesFailedToDeleteFromSink: Bytes that failed to be deleted from the data
      sink.
    bytesFoundFromSource: Bytes found in the data source that are scheduled to
      be transferred, excluding any that are filtered based on object
      conditions or skipped due to sync.
    bytesFoundOnlyFromSink: Bytes found only in the data sink that are
      scheduled to be deleted.
    bytesFromSourceFailed: Bytes in the data source that failed to be
      transferred or that failed to be deleted after being transferred.
    bytesFromSourceSkippedBySync: Bytes in the data source that are not
      transferred because they already exist in the data sink.
    directoriesFailedToListFromSource: For transfers involving PosixFilesystem
      only. Number of listing failures for each directory found at the source.
      Potential failures when listing a directory include permission failure
      or block failure. If listing a directory fails, no files in the
      directory are transferred.
    directoriesFoundFromSource: For transfers involving PosixFilesystem only.
      Number of directories found while listing. For example, if the root
      directory of the transfer is `base/` and there are two other
      directories, `a/` and `b/` under this directory, the count after listing
      `base/`, `base/a/` and `base/b/` is 3.
    directoriesSuccessfullyListedFromSource: For transfers involving
      PosixFilesystem only. Number of successful listings for each directory
      found at the source.
    intermediateObjectsCleanedUp: Number of successfully cleaned up
      intermediate objects.
    intermediateObjectsFailedCleanedUp: Number of intermediate objects failed
      cleaned up.
    objectsCopiedToSink: Objects that are copied to the data sink.
    objectsDeletedFromSink: Objects that are deleted from the data sink.
    objectsDeletedFromSource: Objects that are deleted from the data source.
    objectsFailedToDeleteFromSink: Objects that failed to be deleted from the
      data sink.
    objectsFoundFromSource: Objects found in the data source that are
      scheduled to be transferred, excluding any that are filtered based on
      object conditions or skipped due to sync.
    objectsFoundOnlyFromSink: Objects found only in the data sink that are
      scheduled to be deleted.
    objectsFromSourceFailed: Objects in the data source that failed to be
      transferred or that failed to be deleted after being transferred.
    objectsFromSourceSkippedBySync: Objects in the data source that are not
      transferred because they already exist in the data sink.
  """
    bytesCopiedToSink = _messages.IntegerField(1)
    bytesDeletedFromSink = _messages.IntegerField(2)
    bytesDeletedFromSource = _messages.IntegerField(3)
    bytesFailedToDeleteFromSink = _messages.IntegerField(4)
    bytesFoundFromSource = _messages.IntegerField(5)
    bytesFoundOnlyFromSink = _messages.IntegerField(6)
    bytesFromSourceFailed = _messages.IntegerField(7)
    bytesFromSourceSkippedBySync = _messages.IntegerField(8)
    directoriesFailedToListFromSource = _messages.IntegerField(9)
    directoriesFoundFromSource = _messages.IntegerField(10)
    directoriesSuccessfullyListedFromSource = _messages.IntegerField(11)
    intermediateObjectsCleanedUp = _messages.IntegerField(12)
    intermediateObjectsFailedCleanedUp = _messages.IntegerField(13)
    objectsCopiedToSink = _messages.IntegerField(14)
    objectsDeletedFromSink = _messages.IntegerField(15)
    objectsDeletedFromSource = _messages.IntegerField(16)
    objectsFailedToDeleteFromSink = _messages.IntegerField(17)
    objectsFoundFromSource = _messages.IntegerField(18)
    objectsFoundOnlyFromSink = _messages.IntegerField(19)
    objectsFromSourceFailed = _messages.IntegerField(20)
    objectsFromSourceSkippedBySync = _messages.IntegerField(21)