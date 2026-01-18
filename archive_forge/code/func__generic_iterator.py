import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _generic_iterator(self, file_path):
    """A helper method that makes an iterator given a debug-events file path.

    Repeated calls to this method create iterators that remember the last
    successful reading position (offset) for each given `file_path`. So the
    iterators are meant for incremental reading of the file.

    Args:
      file_path: Path to the file to create the iterator for.

    Yields:
      A tuple of (offset, debug_event_proto) on each `next()` call.
    """
    yield_count = 0
    reader = self._get_reader(file_path)
    read_lock = self._reader_read_locks[file_path]
    read_lock.acquire()
    try:
        while True:
            current_offset = self._reader_offsets[file_path]
            try:
                record, self._reader_offsets[file_path] = reader.read(current_offset)
            except (errors.DataLossError, IndexError):
                break
            yield DebugEventWithOffset(debug_event=debug_event_pb2.DebugEvent.FromString(record), offset=current_offset)
            yield_count += 1
            if yield_count % self._READER_RELEASE_PER == 0:
                read_lock.release()
                read_lock.acquire()
    finally:
        read_lock.release()