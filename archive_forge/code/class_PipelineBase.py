import random
from collections import deque
from datetime import timedelta
from .timer import Timer
class PipelineBase(StatsClientBase):

    def __init__(self, client):
        self._client = client
        self._prefix = client._prefix
        self._stats = deque()

    def _send(self):
        raise NotImplementedError()

    def _after(self, data):
        if data is not None:
            self._stats.append(data)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.send()

    def send(self):
        if not self._stats:
            return
        self._send()

    def pipeline(self):
        return self.__class__(self)