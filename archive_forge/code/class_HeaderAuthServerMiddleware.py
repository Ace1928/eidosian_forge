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
class HeaderAuthServerMiddleware(ServerMiddleware):
    """A ServerMiddleware that transports incoming username and password."""

    def __init__(self, token):
        self.token = token

    def sending_headers(self):
        return {'authorization': 'Bearer ' + self.token}