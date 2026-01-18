import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class TestProto:

    def __init__(self) -> None:
        self._output: List[bytes] = []
        self._received: Dict[int, List[bytes]] = {0: [], 1: [], 2: [], 3: []}

    def set_output(self, output_lines):
        self._output = output_lines

    def read_pkt_line(self):
        if self._output:
            data = self._output.pop(0)
            if data is not None:
                return data.rstrip() + b'\n'
            else:
                return None
        else:
            raise HangupException

    def write_sideband(self, band, data):
        self._received[band].append(data)

    def write_pkt_line(self, data):
        self._received[0].append(data)

    def get_received_line(self, band=0):
        lines = self._received[band]
        return lines.pop(0)