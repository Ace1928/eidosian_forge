import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class LifoTestMixin:

    def test_push_pop1(self):
        """Basic push/pop test"""
        q = self.queue()
        q.push(b'a')
        q.push(b'b')
        q.push(b'c')
        self.assertEqual(q.pop(), b'c')
        self.assertEqual(q.pop(), b'b')
        self.assertEqual(q.pop(), b'a')
        self.assertEqual(q.pop(), None)

    def test_push_pop2(self):
        """Test interleaved push and pops"""
        q = self.queue()
        q.push(b'a')
        q.push(b'b')
        q.push(b'c')
        q.push(b'd')
        self.assertEqual(q.pop(), b'd')
        self.assertEqual(q.pop(), b'c')
        q.push(b'e')
        self.assertEqual(q.pop(), b'e')
        self.assertEqual(q.pop(), b'b')
        self.assertEqual(q.pop(), b'a')

    def test_peek_lifo(self):
        q = self.queue()
        self.assertIsNone(q.peek())
        q.push(b'a')
        q.push(b'b')
        q.push(b'c')
        self.assertEqual(q.peek(), b'c')
        self.assertEqual(q.peek(), b'c')
        self.assertEqual(q.pop(), b'c')
        self.assertEqual(q.peek(), b'b')
        self.assertEqual(q.peek(), b'b')
        self.assertEqual(q.pop(), b'b')
        self.assertEqual(q.peek(), b'a')
        self.assertEqual(q.peek(), b'a')
        self.assertEqual(q.pop(), b'a')
        self.assertIsNone(q.peek())