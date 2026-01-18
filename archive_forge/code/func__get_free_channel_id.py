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
def _get_free_channel_id(self):
    used_channel_ids = set(self.connection._used_channel_ids)
    for channel_id in range(1, self.connection.channel_max + 1):
        if channel_id not in used_channel_ids:
            self.connection._used_channel_ids.append(channel_id)
            return channel_id
    raise ResourceError('No free channel ids, current={}, channel_max={}'.format(len(self.connection.channels), self.connection.channel_max), (20, 10))