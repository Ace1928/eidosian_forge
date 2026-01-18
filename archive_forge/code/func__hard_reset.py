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
def _hard_reset(self, reason):
    """Reset the controller to its pre-connection state"""
    for sender in self._purged_senders:
        sender.destroy(reason)
    del self._purged_senders[:]
    self._active_senders.clear()
    unused = []
    for key, sender in self._all_senders.items():
        if sender.pending_messages == 0:
            unused.append(key)
        else:
            sender.reset(reason)
            self._active_senders.add(key)
    for key in unused:
        self._all_senders[key].destroy(reason)
        del self._all_senders[key]
    for servers in self._servers.values():
        for server in servers.values():
            server.reset()
    if self.reply_link:
        self.reply_link.destroy()
        self.reply_link = None
    if self._socket_connection:
        self._socket_connection.reset()