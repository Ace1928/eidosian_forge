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
class ArbitraryHeadersFlightServer(FlightServerBase):
    """A Flight server that tests multiple arbitrary headers."""

    def do_action(self, context, action):
        middleware = context.get_middleware('arbitrary-headers')
        if middleware:
            headers = middleware.sending_headers()
            header_1 = case_insensitive_header_lookup(headers, 'test-header-1')
            header_2 = case_insensitive_header_lookup(headers, 'test-header-2')
            value1 = header_1[0].encode('utf-8')
            value2 = header_2[0].encode('utf-8')
            return [value1, value2]
        raise flight.FlightServerError('No headers middleware found')