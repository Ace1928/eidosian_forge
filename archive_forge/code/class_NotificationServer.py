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
class NotificationServer(Server):
    """Subscribes to Notification addresses"""

    def __init__(self, target, incoming, scheduler, delay, capacity):
        super(NotificationServer, self).__init__(target, incoming, scheduler, delay, capacity)

    def attach(self, connection, addresser):
        self._addresses = [addresser.anycast_address(self._target, SERVICE_NOTIFY)]
        super(NotificationServer, self).attach(connection)