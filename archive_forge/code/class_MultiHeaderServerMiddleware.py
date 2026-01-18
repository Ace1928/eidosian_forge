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
class MultiHeaderServerMiddleware(ServerMiddleware):
    """Test sending/receiving multiple (binary-valued) headers."""

    def __init__(self, client_headers):
        self.client_headers = client_headers

    def sending_headers(self):
        return MultiHeaderClientMiddleware.EXPECTED