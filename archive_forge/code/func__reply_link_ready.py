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
def _reply_link_ready(self):
    """Invoked when the Replies reply link has become active.  At this
        point, we are ready to receive messages, so start all pending RPC
        requests.
        """
    LOG.info('Messaging is active (%(hostname)s:%(port)s%(vhost)s)', {'hostname': self.hosts.current.hostname, 'port': self.hosts.current.port, 'vhost': '/' + self.hosts.virtual_host if self.hosts.virtual_host else ''})
    for sender in self._all_senders.values():
        sender.attach(self._socket_connection.pyngus_conn, self.reply_link, self.addresser)