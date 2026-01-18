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
class MultiHeaderClientMiddleware(ClientMiddleware):
    """Test sending/receiving multiple (binary-valued) headers."""
    EXPECTED = {'x-text': ['foo', 'bar'], 'x-binary-bin': [b'\x00', b'\x01'], 'x-MIXED-case': ['baz'], b'x-other-MIXED-case': ['baz']}

    def __init__(self, factory):
        self.factory = factory

    def sending_headers(self):
        return self.EXPECTED

    def received_headers(self, headers):
        self.factory.last_headers.update(headers)