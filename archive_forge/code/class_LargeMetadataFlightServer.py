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
class LargeMetadataFlightServer(FlightServerBase):
    """Regression test for ARROW-13253."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = b' ' * (2 ** 31 + 1)

    def do_get(self, context, ticket):
        schema = pa.schema([('a', pa.int64())])
        return flight.GeneratorStream(schema, [(pa.record_batch([[1]], schema=schema), self._metadata)])

    def do_exchange(self, context, descriptor, reader, writer):
        writer.write_metadata(self._metadata)