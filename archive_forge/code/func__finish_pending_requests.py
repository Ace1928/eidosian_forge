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
def _finish_pending_requests(self) -> None:
    """Process any requests that were completed by the last
        call to multi.socket_action.
        """
    while True:
        num_q, ok_list, err_list = self._multi.info_read()
        for curl in ok_list:
            self._finish(curl)
        for curl, errnum, errmsg in err_list:
            self._finish(curl, errnum, errmsg)
        if num_q == 0:
            break
    self._process_queue()