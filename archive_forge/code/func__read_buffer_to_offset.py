import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _read_buffer_to_offset(self, new_buff_offset):
    old_buff_offset = self.buff_offset
    read_size = min(len(self.buff), new_buff_offset) - old_buff_offset
    self.buff_offset += read_size
    return self.buff[old_buff_offset:old_buff_offset + read_size]