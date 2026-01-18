import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def _get_contextvars():
    """
    Figure out the "right" contextvars installation to use. Returns a
    `contextvars.ContextVar`-like class with a limited API.

    See https://docs.sentry.io/platforms/python/contextvars/ for more information.
    """
    if not _is_contextvars_broken():
        if sys.version_info < (3, 7):
            try:
                from aiocontextvars import ContextVar, copy_context
                return (True, ContextVar, copy_context)
            except ImportError:
                pass
        else:
            try:
                from contextvars import ContextVar, copy_context
                return (True, ContextVar, copy_context)
            except ImportError:
                pass
    from threading import local
    return (False, _make_threadlocal_contextvars(local), _make_noop_copy_context())