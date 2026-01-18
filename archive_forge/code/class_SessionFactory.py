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
class SessionFactory(LoggingConfigurable):
    """The Base class for configurables that have a Session, Context, logger,
    and IOLoop.
    """
    logname = Unicode('')

    @observe('logname')
    def _logname_changed(self, change: t.Any) -> None:
        self.log = logging.getLogger(change['new'])
    context = Instance('zmq.Context')

    def _context_default(self) -> zmq.Context:
        return zmq.Context()
    session = Instance('jupyter_client.session.Session', allow_none=True)
    loop = Instance('tornado.ioloop.IOLoop')

    def _loop_default(self) -> IOLoop:
        return IOLoop.current()

    def __init__(self, **kwargs: t.Any) -> None:
        """Initialize a session factory."""
        super().__init__(**kwargs)
        if self.session is None:
            self.session = Session(**kwargs)