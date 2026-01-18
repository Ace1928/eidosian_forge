import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class InterfaceTest(QueuelibTestCase):

    def test_queue(self):
        queue = BaseQueue()
        with self.assertRaises(NotImplementedError):
            queue.push(b'')
        with self.assertRaises(NotImplementedError):
            queue.peek()
        with self.assertRaises(NotImplementedError):
            queue.pop()
        with self.assertRaises(NotImplementedError):
            len(queue)
        queue.close()

    def test_issubclass(self):
        assert not issubclass(list, BaseQueue)
        assert not issubclass(int, BaseQueue)
        assert not issubclass(QueuelibTestCase, BaseQueue)
        assert issubclass(DummyQueue, BaseQueue)
        assert issubclass(FifoMemoryQueue, BaseQueue)
        assert issubclass(LifoMemoryQueue, BaseQueue)
        assert issubclass(FifoDiskQueue, BaseQueue)
        assert issubclass(LifoDiskQueue, BaseQueue)
        assert issubclass(FifoSQLiteQueue, BaseQueue)
        assert issubclass(LifoSQLiteQueue, BaseQueue)

    def test_isinstance(self):
        assert not isinstance(1, BaseQueue)
        assert not isinstance([], BaseQueue)
        assert isinstance(DummyQueue(), BaseQueue)
        assert isinstance(FifoMemoryQueue(), BaseQueue)
        assert isinstance(LifoMemoryQueue(), BaseQueue)
        for cls in [FifoDiskQueue, LifoDiskQueue, FifoSQLiteQueue, LifoSQLiteQueue]:
            queue = cls(self.tempfilename())
            assert isinstance(queue, BaseQueue)
            queue.close()