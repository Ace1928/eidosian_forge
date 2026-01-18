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
class EchoFlightServer(FlightServerBase):
    """A Flight server that returns the last data uploaded."""

    def __init__(self, location=None, expected_schema=None, **kwargs):
        super().__init__(location, **kwargs)
        self.last_message = None
        self.expected_schema = expected_schema

    def do_get(self, context, ticket):
        return flight.RecordBatchStream(self.last_message)

    def do_put(self, context, descriptor, reader, writer):
        if self.expected_schema:
            assert self.expected_schema == reader.schema
        self.last_message = reader.read_all()

    def do_exchange(self, context, descriptor, reader, writer):
        for chunk in reader:
            pass