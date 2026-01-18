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
class ConvenienceServer(FlightServerBase):
    """
    Server for testing various implementation conveniences (auto-boxing, etc.)
    """

    @property
    def simple_action_results(self):
        return [b'foo', b'bar', b'baz']

    def do_action(self, context, action):
        if action.type == 'simple-action':
            return self.simple_action_results
        elif action.type == 'echo':
            return [action.body]
        elif action.type == 'bad-action':
            return ['foo']
        elif action.type == 'arrow-exception':
            raise pa.ArrowMemoryError()
        elif action.type == 'forever':

            def gen():
                while not context.is_cancelled():
                    yield b'foo'
            return gen()