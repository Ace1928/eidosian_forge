import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
class SlowFlightServer(FlightServerBase):
    """A Flight server that delays its responses to test timeouts."""

    def do_get(self, context, ticket):
        return flight.GeneratorStream(pa.schema([('a', pa.int32())]), self.slow_stream())

    def do_action(self, context, action):
        time.sleep(0.5)
        return []

    @staticmethod
    def slow_stream():
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        yield pa.Table.from_arrays(data1, names=['a'])
        time.sleep(10)
        yield pa.Table.from_arrays(data1, names=['a'])