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
class _ListenerThread(threading.Thread):
    """Run a blocking listener in a thread."""

    def __init__(self, listener, msg_count, msg_ack=True):
        super(_ListenerThread, self).__init__()
        self.listener = listener
        self.msg_count = msg_count
        self._msg_ack = msg_ack
        self.messages = queue.Queue()
        self.daemon = True
        self.started = eventletutils.Event()
        self._done = eventletutils.Event()
        self.start()
        self.started.wait()

    def run(self):
        LOG.debug('Listener started')
        self.started.set()
        while not self._done.is_set():
            for in_msg in self.listener.poll(timeout=0.5):
                self.messages.put(in_msg)
                self.msg_count -= 1
                self.msg_count == 0 and self._done.set()
                if self._msg_ack:
                    in_msg.acknowledge()
                    if in_msg.message.get('method') == 'echo':
                        in_msg.reply(reply={'correlation-id': in_msg.message.get('id')})
                else:
                    in_msg.requeue()
        LOG.debug('Listener stopped')

    def get_messages(self):
        """Returns a list of all received messages."""
        msgs = []
        try:
            while True:
                m = self.messages.get(False)
                msgs.append(m)
        except queue.Empty:
            pass
        return msgs

    def kill(self, timeout=30):
        self._done.set()
        self.join(timeout)