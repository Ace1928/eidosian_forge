import collections
import functools
import logging
import pycurl
import threading
import time
from io import BytesIO
from tornado import httputil
from tornado import ioloop
from tornado.escape import utf8, native_str
from tornado.httpclient import (
from tornado.log import app_log
from typing import Dict, Any, Callable, Union, Optional
import typing
def _curl_debug(self, debug_type: int, debug_msg: str) -> None:
    debug_types = ('I', '<', '>', '<', '>')
    if debug_type == 0:
        debug_msg = native_str(debug_msg)
        curl_log.debug('%s', debug_msg.strip())
    elif debug_type in (1, 2):
        debug_msg = native_str(debug_msg)
        for line in debug_msg.splitlines():
            curl_log.debug('%s %s', debug_types[debug_type], line)
    elif debug_type == 4:
        curl_log.debug('%s %r', debug_types[debug_type], debug_msg)