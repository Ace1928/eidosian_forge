from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def _decode_cookie(self, cookie_name, cookie=None):
    from tornado.web import decode_signed_value
    from ..config import config
    cookie = self.cookies.get(cookie_name)
    if cookie is None:
        return None
    cookie = decode_signed_value(config.cookie_secret, cookie_name, cookie)
    return self._decrypt_cookie(cookie)