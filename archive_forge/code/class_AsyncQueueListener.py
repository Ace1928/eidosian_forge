import asyncio
import atexit
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, List, Optional, Tuple, Union
class AsyncQueueListener(QueueListener):

    def __init__(self, queue: queue.Queue, *handlers: logging.Handler, respect_handler_level: bool=False):
        super().__init__(queue, *handlers, respect_handler_level=respect_handler_level)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _monitor(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        super()._monitor()

    def stop(self) -> None:
        if self.loop:
            self.loop.stop()
            self.loop.close()
        self.enqueue_sentinel()
        if self._thread:
            self._thread.join(1)
            self._thread = None