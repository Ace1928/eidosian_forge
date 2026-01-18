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
@observe('signature_scheme')
def _signature_scheme_changed(self, change: t.Any) -> None:
    new = change['new']
    if not new.startswith('hmac-'):
        raise TraitError("signature_scheme must start with 'hmac-', got %r" % new)
    hash_name = new.split('-', 1)[1]
    try:
        self.digest_mod = getattr(hashlib, hash_name)
    except AttributeError as e:
        raise TraitError('hashlib has no such attribute: %s' % hash_name) from e
    self._new_auth()