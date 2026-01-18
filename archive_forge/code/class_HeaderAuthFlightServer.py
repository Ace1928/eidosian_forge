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
class HeaderAuthFlightServer(FlightServerBase):
    """A Flight server that tests with basic token authentication. """

    def do_action(self, context, action):
        middleware = context.get_middleware('auth')
        if middleware:
            auth_header = case_insensitive_header_lookup(middleware.sending_headers(), 'Authorization')
            values = auth_header.split(' ')
            return [values[1].encode('utf-8')]
        raise flight.FlightUnauthenticatedError('No token auth middleware found.')