import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
@property
def _next_deadline(self):
    """The timestamp of the next expiring event or None
        """
    return self._deadlines[0] if self._deadlines else None