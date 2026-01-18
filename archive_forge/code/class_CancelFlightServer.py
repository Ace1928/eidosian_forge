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
class CancelFlightServer(FlightServerBase):
    """A server for testing StopToken."""

    def do_get(self, context, ticket):
        schema = pa.schema([])
        rb = pa.RecordBatch.from_arrays([], schema=schema)
        return flight.GeneratorStream(schema, itertools.repeat(rb))

    def do_exchange(self, context, descriptor, reader, writer):
        schema = pa.schema([])
        rb = pa.RecordBatch.from_arrays([], schema=schema)
        writer.begin(schema)
        while not context.is_cancelled():
            writer.write_batch(rb)
            time.sleep(0.5)