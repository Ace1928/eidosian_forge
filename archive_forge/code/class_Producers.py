from __future__ import annotations
import os
from itertools import chain
from .connection import Resource
from .messaging import Producer
from .utils.collections import EqualityDict
from .utils.compat import register_after_fork
from .utils.functional import lazy
class Producers(PoolGroup):
    """Collection of producer pools."""

    def create(self, connection, limit):
        return ProducerPool(connections[connection], limit=limit)