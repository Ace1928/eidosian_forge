import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
def _schedule_task_processing(self):
    """_process_tasks() helper: prevent queuing up multiple requests for
        task processing.  This method is called both by the application thread
        and the processing thread.
        """
    if self.processor:
        with self._process_tasks_lock:
            already_scheduled = self._process_tasks_scheduled
            self._process_tasks_scheduled = True
        if not already_scheduled:
            self.processor.wakeup(lambda: self._process_tasks())