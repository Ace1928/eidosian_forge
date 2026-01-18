import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Text
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from tornado.httputil import HTTPHeaders
from tornado.ioloop import IOLoop
from tornado.queues import Queue
from traitlets import Float, Instance, default
from traitlets.config import LoggingConfigurable
from .non_blocking import make_non_blocking
@default('max_wait')
def _default_max_wait(self):
    return 0.1 if os.name == 'nt' else self.min_wait * 2