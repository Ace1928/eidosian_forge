import io
import logging
import os
import sys
import threading
import time
from io import StringIO
def _streamReader(self, handle):
    while True:
        new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
        if not new_data:
            break
        handle.decoder_buffer += new_data
        handle.decodeIncomingBuffer()
        handle.writeOutputBuffer(self.ostreams)