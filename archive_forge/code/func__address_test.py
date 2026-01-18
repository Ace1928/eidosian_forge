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
def _address_test(self, rpc_target, targets_priorities):
    broker = FakeBroker(self.conf.oslo_messaging_amqp)
    broker.start()
    url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d' % (broker.host, broker.port))
    driver = amqp_driver.ProtonDriver(self.conf, url)
    rl = []
    for server in ['Server1', 'Server2']:
        _ = driver.listen(rpc_target(server=server), None, None)._poll_style_listener
        rl.append(_ListenerThread(_, 3))
    nl = []
    for n in range(2):
        _ = driver.listen_for_notifications(targets_priorities, None, None, None)._poll_style_listener
        nl.append(_ListenerThread(_, len(targets_priorities)))
    driver.send(rpc_target(server='Server1'), {'context': 'whatever'}, {'msg': 'Server1'})
    driver.send(rpc_target(server='Server2'), {'context': 'whatever'}, {'msg': 'Server2'})
    driver.send(rpc_target(fanout=True), {'context': 'whatever'}, {'msg': 'Fanout'})
    driver.send(rpc_target(server=None), {'context': 'whatever'}, {'msg': 'Anycast1'})
    driver.send(rpc_target(server=None), {'context': 'whatever'}, {'msg': 'Anycast2'})
    expected = []
    for n in targets_priorities:
        topic = '%s.%s' % (n[0].topic, n[1])
        target = oslo_messaging.Target(topic=topic)
        driver.send_notification(target, {'context': 'whatever'}, {'msg': topic}, 2.0)
        expected.append(topic)
    for li in rl:
        li.join(timeout=30)
    predicate = lambda: len(expected) == nl[0].messages.qsize() + nl[1].messages.qsize()
    _wait_until(predicate, 30)
    for li in nl:
        li.kill(timeout=30)
    s1_payload = [m.message.get('msg') for m in rl[0].get_messages()]
    s2_payload = [m.message.get('msg') for m in rl[1].get_messages()]
    self.assertTrue('Server1' in s1_payload and 'Server2' not in s1_payload)
    self.assertTrue('Server2' in s2_payload and 'Server1' not in s2_payload)
    self.assertEqual(s1_payload.count('Fanout'), 1)
    self.assertEqual(s2_payload.count('Fanout'), 1)
    self.assertEqual((s1_payload + s2_payload).count('Anycast1'), 1)
    self.assertEqual((s1_payload + s2_payload).count('Anycast2'), 1)
    n1_payload = [m.message.get('msg') for m in nl[0].get_messages()]
    n2_payload = [m.message.get('msg') for m in nl[1].get_messages()]
    self.assertEqual((n1_payload + n2_payload).sort(), expected.sort())
    driver.cleanup()
    broker.stop()
    return broker.message_log