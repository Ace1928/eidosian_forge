import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class FifoDiskQueueTest(FifoTestMixin, PersistentTestMixin, QueueTestMixin, QueuelibTestCase):

    def queue(self):
        return FifoDiskQueue(self.qpath, chunksize=self.chunksize)

    def test_not_szhdr(self):
        q = self.queue()
        q.push(b'something')
        empty_file = open(self.tempfilename(), 'w+')
        with mock.patch.object(q, 'tailf', empty_file):
            assert q.peek() is None
            assert q.pop() is None
        empty_file.close()

    def test_chunks(self):
        """Test chunks are created and removed"""
        values = [b'0', b'1', b'2', b'3', b'4']
        q = self.queue()
        for x in values:
            q.push(x)
        chunks = glob.glob(os.path.join(self.qpath, 'q*'))
        self.assertEqual(len(chunks), 5 // self.chunksize + 1)
        for x in values:
            q.pop()
        chunks = glob.glob(os.path.join(self.qpath, 'q*'))
        self.assertEqual(len(chunks), 1)