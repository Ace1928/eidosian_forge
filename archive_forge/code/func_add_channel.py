import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def add_channel(self, map=None):
    if map is None:
        map = self._map
    map[self._fileno] = self