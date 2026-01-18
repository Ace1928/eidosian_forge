from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import six
from gslib.exception import CommandException
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.constants import UTF8
class ResumableStreamingJsonUploadWrapper(object):
    """Wraps an input stream in a buffer for resumable uploads.

  This class takes a non-seekable input stream, buffers it, and exposes it
  as a stream with limited seek capabilities such that it can be used in a
  resumable JSON API upload.

  max_buffer_size bytes of buffering is supported.
  """

    def __init__(self, stream, max_buffer_size, test_small_buffer=False):
        """Initializes the wrapper.

    Args:
      stream: Input stream.
      max_buffer_size: Maximum size of internal buffer; should be >= the chunk
          size of the resumable upload API to ensure that at least one full
          chunk write can be replayed in the event of a server error.
      test_small_buffer: Skip check for buffer size vs. chunk size, for testing.
    """
        self._orig_fp = stream
        if not test_small_buffer and max_buffer_size < GetJsonResumableChunkSize():
            raise CommandException('Resumable streaming upload created with buffer size %s, JSON resumable upload chunk size %s. Buffer size must be >= JSON resumable upload chunk size to ensure that uploads can be resumed.' % (max_buffer_size, GetJsonResumableChunkSize()))
        self._max_buffer_size = max_buffer_size
        self._buffer = collections.deque()
        self._buffer_start = 0
        self._buffer_end = 0
        self._position = 0

    @property
    def mode(self):
        """Returns the mode of the underlying file descriptor, or None."""
        return getattr(self._orig_fp, 'mode', None)

    def read(self, size=-1):
        """"Reads from the wrapped stream.

    Args:
      size: The amount of bytes to read. If omitted or negative, the entire
          contents of the stream will be read and returned.

    Returns:
      Bytes from the wrapped stream.
    """
        read_all_bytes = size is None or size < 0
        if read_all_bytes:
            bytes_remaining = self._max_buffer_size
        else:
            bytes_remaining = size
        data = b''
        buffered_data = []
        if self._position < self._buffer_end:
            pos_in_buffer = self._buffer_start
            buffer_index = 0
            while pos_in_buffer + len(self._buffer[buffer_index]) < self._position:
                pos_in_buffer += len(self._buffer[buffer_index])
                buffer_index += 1
            while pos_in_buffer < self._buffer_end and bytes_remaining > 0:
                buffer_len = len(self._buffer[buffer_index])
                offset_from_position = self._position - pos_in_buffer
                bytes_available_this_buffer = buffer_len - offset_from_position
                read_size = min(bytes_available_this_buffer, bytes_remaining)
                buffered_data.append(self._buffer[buffer_index][offset_from_position:offset_from_position + read_size])
                bytes_remaining -= read_size
                pos_in_buffer += buffer_len
                buffer_index += 1
                self._position += read_size
        if read_all_bytes:
            new_data = self._orig_fp.read(size)
            data_len = len(new_data)
            if not buffered_data:
                data = new_data
            else:
                buffered_data.append(new_data)
                data = b''.join(buffered_data)
            self._position += data_len
        elif bytes_remaining:
            new_data = self._orig_fp.read(bytes_remaining)
            if not buffered_data:
                data = new_data
            else:
                buffered_data.append(new_data)
                data = b''.join(buffered_data)
            data_len = len(new_data)
            if data_len:
                self._position += data_len
                self._buffer.append(new_data)
                self._buffer_end += data_len
                oldest_data = None
                while self._buffer_end - self._buffer_start > self._max_buffer_size:
                    oldest_data = self._buffer.popleft()
                    self._buffer_start += len(oldest_data)
                if oldest_data:
                    refill_amount = self._max_buffer_size - (self._buffer_end - self._buffer_start)
                    if refill_amount:
                        self._buffer.appendleft(oldest_data[-refill_amount:])
                        self._buffer_start -= refill_amount
        else:
            if six.PY3:
                if buffered_data:
                    buffered_data = [bd.encode(UTF8) if isinstance(bd, str) else bd for bd in buffered_data]
            data = b''.join(buffered_data) if buffered_data else b''
        return data

    def tell(self):
        """Returns the current stream position."""
        return self._position

    def seekable(self):
        """Returns true since limited seek support exists."""
        return True

    def seek(self, offset, whence=os.SEEK_SET):
        """Seeks on the buffered stream.

    Args:
      offset: The offset to seek to; must be within the buffer bounds.
      whence: Must be os.SEEK_SET.

    Raises:
      CommandException if an unsupported seek mode or position is used.
    """
        if whence == os.SEEK_SET:
            if offset < self._buffer_start or offset > self._buffer_end:
                raise CommandException('Unable to resume upload because of limited buffering available for streaming uploads. Offset %s was requested, but only data from %s to %s is buffered.' % (offset, self._buffer_start, self._buffer_end))
            self._position = offset
        elif whence == os.SEEK_END:
            if offset > self._max_buffer_size:
                raise CommandException('Invalid SEEK_END offset %s on streaming upload. Only %s can be buffered.' % (offset, self._max_buffer_size))
            while self.read(self._max_buffer_size):
                pass
            self._position -= offset
        else:
            raise CommandException('Invalid seek mode on streaming upload. (mode %s, offset %s)' % (whence, offset))

    def close(self):
        return self._orig_fp.close()