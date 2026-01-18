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
class RPCMonitoredCallTask(RPCCallTask):
    """An RPC call which expects a periodic heartbeat until the response is
    received.  There are two timeouts:
    deadline - overall hard timeout, implemented in RPCCallTask
    call_monitor_timeout - keep alive timeout, reset when heartbeat arrives
    """

    def __init__(self, target, message, deadline, call_monitor_timeout, retry, wait_for_ack):
        super(RPCMonitoredCallTask, self).__init__(target, message, deadline, retry, wait_for_ack)
        assert call_monitor_timeout is not None
        self._monitor_timeout = call_monitor_timeout
        self._monitor_timer = None
        self._set_alarm = None

    def _execute(self, controller):
        self._set_alarm = controller.processor.defer
        self._monitor_timer = self._set_alarm(self._call_timeout, self._monitor_timeout)
        super(RPCMonitoredCallTask, self)._execute(controller)

    def _call_timeout(self):
        self._monitor_timer = None
        self._sender and self._sender.cancel_send(self)
        msg = '{name} message sent to {target} failed: call monitor timed out'.format(name=self.name, target=self.target)
        LOG.warning('%s', msg)
        self._error = exceptions.MessagingTimeout(msg)
        self._cleanup()
        self._wakeup.set()

    def _on_reply(self, message):
        if message.body is None:
            self._monitor_timer.cancel()
            self._monitor_timer = self._set_alarm(self._call_timeout, self._monitor_timeout)
        else:
            super(RPCMonitoredCallTask, self)._on_reply(message)

    def _cleanup(self):
        self._set_alarm = None
        if self._monitor_timer:
            self._monitor_timer.cancel()
            self._monitor_timer = None
        super(RPCMonitoredCallTask, self)._cleanup()