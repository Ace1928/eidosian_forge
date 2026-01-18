import logging
import os
import queue
import threading
import time
import uuid
import cachetools
from oslo_concurrency import lockutils
from oslo_utils import eventletutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging import MessageDeliveryFailure
class MessageOperationsHandler(object):
    """Queue used by message operations to ensure that all tasks are
    serialized and run in the same thread, since underlying drivers like kombu
    are not thread safe.
    """

    def __init__(self, name):
        self.name = '%s (%s)' % (name, hex(id(self)))
        self._tasks = queue.Queue()
        self._shutdown = eventletutils.Event()
        self._shutdown_thread = threading.Thread(target=self._process_in_background)
        self._shutdown_thread.daemon = True

    def stop(self):
        self._shutdown.set()

    def process_in_background(self):
        """Run all pending tasks queued by do() in an thread during the
        shutdown process.
        """
        self._shutdown_thread.start()

    def _process_in_background(self):
        while not self._shutdown.is_set():
            self.process()
            time.sleep(ACK_REQUEUE_EVERY_SECONDS_MIN)

    def process(self):
        """Run all pending tasks queued by do()."""
        while True:
            try:
                task = self._tasks.get(block=False)
            except queue.Empty:
                break
            task()

    def do(self, task):
        """Put the task in the queue."""
        self._tasks.put(task)