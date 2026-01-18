from __future__ import annotations
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest
import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates
def _cull_digest_history(self) -> None:
    """cull the digest history

        Removes a randomly selected 10% of the digest history
        """
    current = len(self.digest_history)
    n_to_cull = max(int(current // 10), current - self.digest_history_size)
    if n_to_cull >= current:
        self.digest_history = set()
        return
    to_cull = random.sample(tuple(sorted(self.digest_history)), n_to_cull)
    self.digest_history.difference_update(to_cull)