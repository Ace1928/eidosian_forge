from __future__ import annotations
import os
from itertools import chain
from .connection import Resource
from .messaging import Producer
from .utils.collections import EqualityDict
from .utils.compat import register_after_fork
from .utils.functional import lazy
class ProducerPool(Resource):
    """Pool of :class:`kombu.Producer` instances."""
    Producer = Producer
    close_after_fork = True

    def __init__(self, connections, *args, **kwargs):
        self.connections = connections
        self.Producer = kwargs.pop('Producer', None) or self.Producer
        super().__init__(*args, **kwargs)

    def _acquire_connection(self):
        return self.connections.acquire(block=True)

    def create_producer(self):
        conn = self._acquire_connection()
        try:
            return self.Producer(conn)
        except BaseException:
            conn.release()
            raise

    def new(self):
        return lazy(self.create_producer)

    def setup(self):
        if self.limit:
            for _ in range(self.limit):
                self._resource.put_nowait(self.new())

    def close_resource(self, resource):
        pass

    def prepare(self, p):
        if callable(p):
            p = p()
        if p._channel is None:
            conn = self._acquire_connection()
            try:
                p.revive(conn)
            except BaseException:
                conn.release()
                raise
        return p

    def release(self, resource):
        if resource.__connection__:
            resource.__connection__.release()
        resource.channel = None
        super().release(resource)