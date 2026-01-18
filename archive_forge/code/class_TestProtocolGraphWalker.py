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
class TestProtocolGraphWalker:

    def __init__(self) -> None:
        self.acks: List[bytes] = []
        self.lines: List[bytes] = []
        self.wants_satisified = False
        self.stateless_rpc = None
        self.advertise_refs = False
        self._impl = None
        self.done_required = True
        self.done_received = False
        self._empty = False
        self.pack_sent = False

    def read_proto_line(self, allowed):
        command, sha = self.lines.pop(0)
        if allowed is not None:
            assert command in allowed
        return (command, sha)

    def send_ack(self, sha, ack_type=b''):
        self.acks.append((sha, ack_type))

    def send_nak(self):
        self.acks.append((None, b'nak'))

    def all_wants_satisfied(self, haves):
        if haves:
            return self.wants_satisified

    def pop_ack(self):
        if not self.acks:
            return None
        return self.acks.pop(0)

    def handle_done(self):
        if not self._impl:
            return
        self.pack_sent = self._impl.handle_done(self.done_required, self.done_received)
        return self.pack_sent

    def notify_done(self):
        self.done_received = True