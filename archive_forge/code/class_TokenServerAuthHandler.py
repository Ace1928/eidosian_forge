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
class TokenServerAuthHandler(ServerAuthHandler):
    """An example implementation of authentication via handshake."""

    def __init__(self, creds):
        super().__init__()
        self.creds = creds

    def authenticate(self, outgoing, incoming):
        username = incoming.read()
        password = incoming.read()
        if username in self.creds and self.creds[username] == password:
            outgoing.write(base64.b64encode(b'secret:' + username))
        else:
            raise flight.FlightUnauthenticatedError('invalid username/password')

    def is_valid(self, token):
        token = base64.b64decode(token)
        if not token.startswith(b'secret:'):
            raise flight.FlightUnauthenticatedError('invalid token')
        return token[7:]