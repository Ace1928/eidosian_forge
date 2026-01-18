import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def _write_data(self, s):
    start_offset = self._index
    offset = self._index % LEVELDBLOG_BLOCK_LEN
    space_left = LEVELDBLOG_BLOCK_LEN - offset
    data_used = 0
    data_left = len(s)
    if space_left < LEVELDBLOG_HEADER_LEN:
        pad = '\x00' * space_left
        self._fp.write(strtobytes(pad))
        self._index += space_left
        offset = 0
        space_left = LEVELDBLOG_BLOCK_LEN
    if data_left + LEVELDBLOG_HEADER_LEN <= space_left:
        self._write_record(s)
    else:
        data_room = space_left - LEVELDBLOG_HEADER_LEN
        self._write_record(s[:data_room], LEVELDBLOG_FIRST)
        data_used += data_room
        data_left -= data_room
        assert data_left, 'data_left should be non-zero'
        while data_left > LEVELDBLOG_DATA_LEN:
            self._write_record(s[data_used:data_used + LEVELDBLOG_DATA_LEN], LEVELDBLOG_MIDDLE)
            data_used += LEVELDBLOG_DATA_LEN
            data_left -= LEVELDBLOG_DATA_LEN
        self._write_record(s[data_used:], LEVELDBLOG_LAST)
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._flush_offset = self._index
    return (start_offset, self._index, self._flush_offset)