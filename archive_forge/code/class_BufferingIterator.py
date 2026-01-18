import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
class BufferingIterator(_bufferingIterator_Base[T], Generic[T]):

    def __init__(self, stream):
        self._stream = iter(stream)
        self._buffer = collections.deque()
        self._expired = False

    def __next__(self):
        if self._buffer:
            return self._buffer.popleft()
        if self._expired:
            raise StopIteration
        return next(self._stream)

    def takewhile(self, predicate):
        """Variant of itertools.takewhile except it does not discard the first non-matching token"""
        buffer = self._buffer
        while buffer or self._fill_buffer(5):
            v = buffer[0]
            if predicate(v):
                buffer.popleft()
                yield v
            else:
                break

    def consume_many(self, count):
        self._fill_buffer(count)
        buffer = self._buffer
        if len(buffer) == count:
            ret = list(buffer)
            buffer.clear()
        else:
            ret = []
            while buffer and count:
                ret.append(buffer.popleft())
                count -= 1
        return ret

    def peek_buffer(self):
        return list(self._buffer)

    def peek_find(self, predicate, limit=None):
        buffer = self._buffer
        i = 0
        while limit is None or i < limit:
            if i >= len(buffer):
                self._fill_buffer(i + 5)
                if i >= len(buffer):
                    return None
            v = buffer[i]
            if predicate(v):
                return i + 1
            i += 1
        return None

    def _fill_buffer(self, number):
        if not self._expired:
            while len(self._buffer) < number:
                try:
                    self._buffer.append(next(self._stream))
                except StopIteration:
                    self._expired = True
                    break
        return bool(self._buffer)

    def peek(self):
        return self.peek_at(1)

    def peek_at(self, tokens_ahead):
        self._fill_buffer(tokens_ahead)
        return self._buffer[tokens_ahead - 1] if len(self._buffer) >= tokens_ahead else None

    def peek_many(self, number):
        self._fill_buffer(number)
        buffer = self._buffer
        if len(buffer) == number:
            ret = list(buffer)
        elif number:
            ret = []
            for t in buffer:
                ret.append(t)
                number -= 1
                if not number:
                    break
        else:
            ret = []
        return ret