import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class _CallMonitor(_ListenerThread):

    def __init__(self, listener, delay, hb_count, msg_count=1):
        self._delay = delay
        self._hb_count = hb_count
        super(_CallMonitor, self).__init__(listener, msg_count)

    def run(self):
        LOG.debug('_CallMonitor started')
        self.started.set()
        while not self._done.is_set():
            for in_msg in self.listener.poll(timeout=0.5):
                hb_rate = in_msg.client_timeout / 2.0
                deadline = time.time() + self._delay
                while deadline > time.time():
                    if self._done.wait(hb_rate):
                        return
                    if self._hb_count > 0:
                        in_msg.heartbeat()
                        self._hb_count -= 1
                in_msg.acknowledge()
                in_msg.reply(reply={'correlation-id': in_msg.message.get('id')})
                self.messages.put(in_msg)
                self.msg_count -= 1
                self.msg_count == 0 and self._done.set()