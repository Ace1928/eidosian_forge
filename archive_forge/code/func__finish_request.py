import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def _finish_request(self, future: 'Optional[Future[None]]') -> None:
    self._clear_callbacks()
    if not self.is_client and self._disconnect_on_finish:
        self.close()
        return
    self.stream.set_nodelay(False)
    if not self._finish_future.done():
        future_set_result_unless_cancelled(self._finish_future, None)