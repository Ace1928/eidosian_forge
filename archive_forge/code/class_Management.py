from __future__ import annotations
import base64
import socket
import sys
import warnings
from array import array
from collections import OrderedDict, defaultdict, namedtuple
from itertools import count
from multiprocessing.util import Finalize
from queue import Empty
from time import monotonic, sleep
from typing import TYPE_CHECKING
from amqp.protocol import queue_declare_ok_t
from kombu.exceptions import ChannelError, ResourceError
from kombu.log import get_logger
from kombu.transport import base
from kombu.utils.div import emergency_dump_state
from kombu.utils.encoding import bytes_to_str, str_to_bytes
from kombu.utils.scheduling import FairCycle
from kombu.utils.uuid import uuid
from .exchange import STANDARD_EXCHANGE_TYPES
class Management(base.Management):
    """Base class for the AMQP management API."""

    def __init__(self, transport):
        super().__init__(transport)
        self.channel = transport.client.channel()

    def get_bindings(self):
        return [{'destination': q, 'source': e, 'routing_key': r} for q, e, r in self.channel.list_bindings()]

    def close(self):
        self.channel.close()