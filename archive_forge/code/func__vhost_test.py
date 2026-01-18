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
def _vhost_test(self):
    """Verify that all messaging for a particular vhost stays on that vhost
        """
    self.config(pseudo_vhost=True, group='oslo_messaging_amqp')
    vhosts = ['None', 'HOSTA', 'HOSTB', 'HOSTC']
    target = oslo_messaging.Target(topic='test-topic')
    fanout = oslo_messaging.Target(topic='test-topic', fanout=True)
    listeners = {}
    ldrivers = {}
    sdrivers = {}
    replies = {}
    msgs = {}
    for vhost in vhosts:
        url = copy.copy(self._broker_url)
        url.virtual_host = vhost if vhost != 'None' else None
        ldriver = amqp_driver.ProtonDriver(self.conf, url)
        listeners[vhost] = _ListenerThread(ldriver.listen(target, None, None)._poll_style_listener, 10)
        ldrivers[vhost] = ldriver
        sdrivers[vhost] = amqp_driver.ProtonDriver(self.conf, url)
        replies[vhost] = []
        msgs[vhost] = []
    for vhost in vhosts:
        if vhost == 'HOSTC':
            continue
        sdrivers[vhost].send(fanout, {'context': vhost}, {'vhost': vhost, 'fanout': True, 'id': vhost})
        replies[vhost].append(sdrivers[vhost].send(target, {'context': vhost}, {'method': 'echo', 'id': vhost}, wait_for_reply=True))
    time.sleep(1)
    for vhost in vhosts:
        msgs[vhost] += listeners[vhost].get_messages()
        if vhost == 'HOSTC':
            self.assertEqual(0, len(msgs[vhost]))
            self.assertEqual(0, len(replies[vhost]))
            continue
        self.assertEqual(2, len(msgs[vhost]))
        for m in msgs[vhost]:
            self.assertEqual(vhost, m.message.get('id'))
        self.assertEqual(1, len(replies[vhost]))
        for m in replies[vhost]:
            self.assertEqual(vhost, m.get('correlation-id'))
    for vhost in vhosts:
        listeners[vhost].kill()
        ldrivers[vhost].cleanup
        sdrivers[vhost].cleanup()