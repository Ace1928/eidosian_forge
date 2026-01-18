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
class RPCCallTask(SendTask):
    """Performs an RPC Call.  Sends the request and waits for a response from
    the destination.
    """

    def __init__(self, target, message, deadline, retry, wait_for_ack):
        super(RPCCallTask, self).__init__('RPC Call', message, target, deadline, retry, wait_for_ack)
        self._reply_link = None
        self._reply_msg = None
        self._msg_id = None

    def wait(self):
        error = super(RPCCallTask, self).wait()
        return error or self._reply_msg

    def _prepare(self, sender):
        super(RPCCallTask, self)._prepare(sender)
        if self._msg_id:
            self._reply_link.cancel_response(self._msg_id)
        self._reply_link = sender._reply_link
        rl = self._reply_link
        self._msg_id = rl.prepare_for_response(self.message, self._on_reply)

    def _on_reply(self, message):
        self._reply_msg = message
        self._cleanup()
        self._wakeup.set()

    def _on_ack(self, state, info):
        if state != pyngus.SenderLink.ACCEPTED:
            super(RPCCallTask, self)._on_ack(state, info)

    def _cleanup(self):
        if self._msg_id:
            self._reply_link.cancel_response(self._msg_id)
            self._msg_id = None
        self._reply_link = None
        super(RPCCallTask, self)._cleanup()