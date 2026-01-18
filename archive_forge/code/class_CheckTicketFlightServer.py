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
class CheckTicketFlightServer(FlightServerBase):
    """A Flight server that compares the given ticket to an expected value."""

    def __init__(self, expected_ticket, location=None, **kwargs):
        super().__init__(location, **kwargs)
        self.expected_ticket = expected_ticket

    def do_get(self, context, ticket):
        assert self.expected_ticket == ticket.ticket
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        table = pa.Table.from_arrays(data1, names=['a'])
        return flight.RecordBatchStream(table)

    def do_put(self, context, descriptor, reader):
        self.last_message = reader.read_all()