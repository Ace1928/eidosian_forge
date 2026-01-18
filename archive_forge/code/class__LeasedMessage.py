from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
class _LeasedMessage(typing.NamedTuple):
    sent_time: float
    'The local time when ACK ID was initially leased in seconds since the epoch.'
    size: int
    ordering_key: Optional[str]