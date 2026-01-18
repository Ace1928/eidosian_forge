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
class MultiAckGraphWalkerImplTestCase(AckGraphWalkerImplTestCase):
    impl_cls = MultiAckGraphWalkerImpl

    def test_multi_ack(self):
        self.assertNextEquals(TWO)
        self.assertNoAck()
        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b'continue')
        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b'continue')
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(THREE)

    def test_multi_ack_partial(self):
        self.assertNextEquals(TWO)
        self.assertNoAck()
        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b'continue')
        self.assertNextEquals(THREE)
        self.assertNoAck()
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(ONE)

    def test_multi_ack_flush(self):
        self._walker.lines = [(b'have', TWO), (None, None), (b'have', ONE), (b'have', THREE), (b'done', None)]
        self.assertNextEquals(TWO)
        self.assertNoAck()
        self.assertNextEquals(ONE)
        self.assertNak()
        self._impl.ack(ONE)
        self.assertAck(ONE, b'continue')
        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b'continue')
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(THREE)

    def test_multi_ack_nak(self):
        self.assertNextEquals(TWO)
        self.assertNoAck()
        self.assertNextEquals(ONE)
        self.assertNoAck()
        self.assertNextEquals(THREE)
        self.assertNoAck()
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()