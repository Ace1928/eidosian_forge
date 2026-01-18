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
class LspStdIoWriter(LspStdIoBase):
    """Language Server stdio Writer"""

    async def write(self) -> None:
        """Write to a Language Server until it closes"""
        while not self.stream.closed:
            message = await self.queue.get()
            try:
                body = message.encode('utf-8')
                response = 'Content-Length: {}\r\n\r\n{}'.format(len(body), message)
                await convert_yielded(self._write_one(response.encode('utf-8')))
            except Exception:
                self.log.exception("%s couldn't write message: %s", self, response)
            finally:
                self.queue.task_done()

    @run_on_executor
    def _write_one(self, message) -> None:
        self.stream.write(message)
        self.stream.flush()