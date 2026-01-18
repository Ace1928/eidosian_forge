import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class _AutoreleasepoolManager:

    def __init__(self):
        self.current = 0
        self.pools = [None]

    @property
    def count(self):
        """Number of total pools. Not including global."""
        return len(self.pools) - 1

    def create(self, pool):
        self.pools.append(pool)
        self.current = self.pools.index(pool)

    def delete(self, pool):
        self.pools.remove(pool)
        self.current = len(self.pools) - 1