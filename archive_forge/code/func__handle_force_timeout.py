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
def _handle_force_timeout(self) -> None:
    """Called by IOLoop periodically to ask libcurl to process any
        events it may have forgotten about.
        """
    while True:
        try:
            ret, num_handles = self._multi.socket_all()
        except pycurl.error as e:
            ret = e.args[0]
        if ret != pycurl.E_CALL_MULTI_PERFORM:
            break
    self._finish_pending_requests()