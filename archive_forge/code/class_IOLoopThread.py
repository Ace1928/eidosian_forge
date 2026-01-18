import asyncio
import atexit
import time
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional
import zmq
from tornado.ioloop import IOLoop
from traitlets import Instance, Type
from traitlets.log import get_logger
from zmq.eventloop import zmqstream
from .channels import HBChannel
from .client import KernelClient
from .session import Session
class IOLoopThread(Thread):
    """Run a pyzmq ioloop in a thread to send and receive messages"""
    _exiting = False
    ioloop = None

    def __init__(self) -> None:
        """Initialize an io loop thread."""
        super().__init__()
        self.daemon = True

    @staticmethod
    @atexit.register
    def _notice_exit() -> None:
        if IOLoopThread is not None:
            IOLoopThread._exiting = True

    def start(self) -> None:
        """Start the IOLoop thread

        Don't return until self.ioloop is defined,
        which is created in the thread
        """
        self._start_future: Future = Future()
        Thread.start(self)
        self._start_future.result(timeout=10)

    def run(self) -> None:
        """Run my loop, ignoring EINTR events in the poller"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def assign_ioloop() -> None:
                self.ioloop = IOLoop.current()
            loop.run_until_complete(assign_ioloop())
        except Exception as e:
            self._start_future.set_exception(e)
        else:
            self._start_future.set_result(None)
        loop.run_until_complete(self._async_run())

    async def _async_run(self) -> None:
        """Run forever (until self._exiting is set)"""
        while not self._exiting:
            await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the channel's event loop and join its thread.

        This calls :meth:`~threading.Thread.join` and returns when the thread
        terminates. :class:`RuntimeError` will be raised if
        :meth:`~threading.Thread.start` is called again.
        """
        self._exiting = True
        self.join()
        self.close()
        self.ioloop = None

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the io loop thread."""
        if self.ioloop is not None:
            try:
                self.ioloop.close(all_fds=True)
            except Exception:
                pass