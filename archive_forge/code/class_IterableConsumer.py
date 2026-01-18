import logging
import threading
from grpc.framework.foundation import stream
class IterableConsumer(stream.Consumer):
    """A Consumer that when iterated over emits the values it has consumed."""

    def __init__(self):
        self._condition = threading.Condition()
        self._values = []
        self._active = True

    def consume(self, value):
        with self._condition:
            if self._active:
                self._values.append(value)
                self._condition.notify()

    def terminate(self):
        with self._condition:
            self._active = False
            self._condition.notify()

    def consume_and_terminate(self, value):
        with self._condition:
            if self._active:
                self._values.append(value)
                self._active = False
                self._condition.notify()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        with self._condition:
            while self._active and (not self._values):
                self._condition.wait()
            if self._values:
                return self._values.pop(0)
            else:
                raise StopIteration()