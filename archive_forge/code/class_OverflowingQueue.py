import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
class OverflowingQueue(queue.Queue):

    def put(self, item, block=True, timeout=None):
        try:
            return queue.Queue.put(self, item, block, timeout)
        except queue.Full:
            pass
        return None

    def put_nowait(self, item):
        return self.put(item, False)