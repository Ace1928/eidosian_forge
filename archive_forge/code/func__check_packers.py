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
def _check_packers(self) -> None:
    """check packers for datetime support."""
    pack = self.pack
    unpack = self.unpack
    msg_list = {'a': [1, 'hi']}
    try:
        packed = pack(msg_list)
    except Exception as e:
        msg = f"packer '{self.packer}' could not serialize a simple message: {e}"
        raise ValueError(msg) from e
    if not isinstance(packed, bytes):
        raise ValueError('message packed to %r, but bytes are required' % type(packed))
    try:
        unpacked = unpack(packed)
        assert unpacked == msg_list
    except Exception as e:
        msg = f"unpacker '{self.unpacker}' could not handle output from packer '{self.packer}': {e}"
        raise ValueError(msg) from e
    msg_datetime = {'t': utcnow()}
    try:
        unpacked = unpack(pack(msg_datetime))
        if isinstance(unpacked['t'], datetime):
            msg = "Shouldn't deserialize to datetime"
            raise ValueError(msg)
    except Exception:
        self.pack = lambda o: pack(squash_dates(o))
        self.unpack = lambda s: unpack(s)