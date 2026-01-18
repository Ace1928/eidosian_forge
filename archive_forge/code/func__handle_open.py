import os
import sys
import uuid
import traceback
import json
import param
from ._version import __version__
def _handle_open(self, comm, msg):
    self._comm = comm
    self._comm.on_msg(self._handle_msg)
    if self._on_open:
        self._on_open(msg)