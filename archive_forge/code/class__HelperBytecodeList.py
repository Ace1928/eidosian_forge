from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break
class _HelperBytecodeList(object):
    """
    A helper double-linked list to make the manipulation a bit easier (so that we don't need
    to keep track of indices that change) and performant (because adding multiple items to
    the middle of a regular list isn't ideal).
    """

    def __init__(self, lst=None):
        self._head = None
        self._tail = None
        if lst:
            node = self
            for item in lst:
                node = node.append(item)

    def append(self, data):
        if self._tail is None:
            node = _Node(data)
            self._head = self._tail = node
            return node
        else:
            node = self._tail = self.tail.append(data)
            return node

    @property
    def head(self):
        node = self._head
        while node.prev:
            self._head = node = node.prev
        return node

    @property
    def tail(self):
        node = self._tail
        while node.next:
            self._tail = node = node.next
        return node

    def __iter__(self):
        node = self.head
        while node:
            yield node.data
            node = node.next