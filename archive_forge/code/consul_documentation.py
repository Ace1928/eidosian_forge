from __future__ import annotations
import socket
import uuid
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from time import monotonic
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
Get the first available message from the queue.

        Before it does so it acquires a lock on the Key/Value store so
        only one node reads at the same time. This is for read consistency
        