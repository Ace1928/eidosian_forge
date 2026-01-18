import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _consume_data_from_channel(self, chan, recv_method, recv_ready_method):
    """
        Try to consume data from the provided channel.

        Keep in mind that data is only consumed if the channel is receive
        ready.
        """
    result = StringIO()
    result_bytes = bytearray()
    if recv_ready_method():
        data = recv_method(self.CHUNK_SIZE)
        result_bytes += b(data)
        while data:
            ready = recv_ready_method()
            if not ready:
                break
            data = recv_method(self.CHUNK_SIZE)
            result_bytes += b(data)
    result.write(result_bytes.decode('utf-8', errors='ignore'))
    return result