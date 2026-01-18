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
def _purge_sender_links(self):
    """Purge inactive sender links"""
    if not self._closing:
        for sender in self._purged_senders:
            sender.destroy('Idle link purged')
        del self._purged_senders[:]
        purge = set(self._all_senders.keys()) - self._active_senders
        for key in purge:
            sender = self._all_senders[key]
            if not sender.pending_messages and (not sender.unacked_messages):
                sender.detach()
                self._purged_senders.append(self._all_senders.pop(key))
        self._active_senders.clear()
        self._link_maint_timer = self.processor.defer(self._purge_sender_links, self._link_maint_timeout)