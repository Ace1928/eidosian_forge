from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import upload_stream
def _store_data(self, data):
    """Adds data to the buffer, respecting max_buffer_size.

    The buffer can consist of many different blocks of data, e.g.

      [b'0', b'12', b'3']

    With a maximum size of 4, if we read two bytes, we must discard the oldest
    data and keep half of the second-oldest block:

      [b'2', b'3', b'45']

    Args:
      data (bytes): the data being added to the buffer.
    """
    if data:
        self._buffer.append(data)
        self._buffer_end += len(data)
        oldest_data = None
        while self._buffer_end - self._buffer_start > self._max_buffer_size:
            oldest_data = self._buffer.popleft()
            self._buffer_start += len(oldest_data)
            if oldest_data:
                refill_amount = self._max_buffer_size - (self._buffer_end - self._buffer_start)
                if refill_amount >= 1:
                    self._buffer.appendleft(oldest_data[-refill_amount:])
                    self._buffer_start -= refill_amount
                hash_util.update_digesters(self._checkpoint_digesters, oldest_data[:len(oldest_data) - refill_amount])
                self._checkpoint_absolute_index = self._buffer_start