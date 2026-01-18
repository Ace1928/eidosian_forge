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
def _send_retry(self, reject, retries):
    self._reject = reject

    def on_active(link):
        if self._reject > 0:
            link.close()
            self._reject -= 1
        else:
            link.add_capacity(10)
    self._broker.on_receiver_active = on_active
    self._broker.start()
    self.config(link_retry_delay=1, group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    try:
        rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'e1'}, wait_for_reply=True, retry=retries)
        self.assertIsNotNone(rc)
        self.assertEqual(rc.get('correlation-id'), 'e1')
    except Exception:
        listener.kill()
        driver.cleanup()
        raise
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    self.assertEqual(listener.messages.get().message.get('method'), 'echo')
    driver.cleanup()