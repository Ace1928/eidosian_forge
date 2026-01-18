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
class HeaderFlightServer(FlightServerBase):
    """Echo back the per-call hard-coded value."""

    def do_action(self, context, action):
        middleware = context.get_middleware('test')
        if middleware:
            return [middleware.special_value.encode()]
        return [b'']