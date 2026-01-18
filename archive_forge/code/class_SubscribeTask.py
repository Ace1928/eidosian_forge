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
class SubscribeTask(Task):
    """A task that creates a subscription to the given target.  Messages
    arriving from the target are given to the listener.
    """

    def __init__(self, target, listener, notifications=False):
        super(SubscribeTask, self).__init__()
        self._target = target()
        self._subscriber_id = listener.id
        self._in_queue = listener.incoming
        self._service = SERVICE_NOTIFY if notifications else SERVICE_RPC
        self._wakeup = eventletutils.Event()

    def wait(self):
        self._wakeup.wait()

    def _execute(self, controller):
        controller.subscribe(self)
        self._wakeup.set()