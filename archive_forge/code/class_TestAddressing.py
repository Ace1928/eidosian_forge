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
@testtools.skipUnless(pyngus, 'proton modules not present')
class TestAddressing(test_utils.BaseTestCase):

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

    def test_routable_address(self):
        self.config(addressing_mode='routable', group='oslo_messaging_amqp')
        _opts = self.conf.oslo_messaging_amqp
        notifications = [(oslo_messaging.Target(topic='test-topic'), 'info'), (oslo_messaging.Target(topic='test-topic'), 'error'), (oslo_messaging.Target(topic='test-topic'), 'debug')]
        msgs = self._address_test(oslo_messaging.Target(exchange='ex', topic='test-topic'), notifications)
        addrs = [m.address for m in msgs]
        notify_addrs = [a for a in addrs if a.startswith(_opts.notify_address_prefix)]
        self.assertEqual(len(notify_addrs), len(notifications))
        self.assertEqual(len(notifications), len([a for a in notify_addrs if _opts.anycast_address in a]))
        rpc_addrs = [a for a in addrs if a.startswith(_opts.rpc_address_prefix)]
        self.assertEqual(2, len([a for a in rpc_addrs if _opts.anycast_address in a]))
        self.assertEqual(1, len([a for a in rpc_addrs if _opts.multicast_address in a]))
        self.assertEqual(2, len([a for a in rpc_addrs if _opts.unicast_address in a]))

    def test_legacy_address(self):
        self.config(addressing_mode='legacy', group='oslo_messaging_amqp')
        _opts = self.conf.oslo_messaging_amqp
        notifications = [(oslo_messaging.Target(topic='test-topic'), 'info'), (oslo_messaging.Target(topic='test-topic'), 'error'), (oslo_messaging.Target(topic='test-topic'), 'debug')]
        msgs = self._address_test(oslo_messaging.Target(exchange='ex', topic='test-topic'), notifications)
        addrs = [m.address for m in msgs]
        server_addrs = [a for a in addrs if a.startswith(_opts.server_request_prefix)]
        broadcast_addrs = [a for a in addrs if a.startswith(_opts.broadcast_prefix)]
        group_addrs = [a for a in addrs if a.startswith(_opts.group_request_prefix)]
        self.assertEqual(len(server_addrs), 2)
        self.assertEqual(len(broadcast_addrs), 1)
        self.assertEqual(len(group_addrs), 2 + len(notifications))

    def test_address_options(self):
        self.config(addressing_mode='routable', group='oslo_messaging_amqp')
        self.config(rpc_address_prefix='RPC-PREFIX', group='oslo_messaging_amqp')
        self.config(notify_address_prefix='NOTIFY-PREFIX', group='oslo_messaging_amqp')
        self.config(multicast_address='MULTI-CAST', group='oslo_messaging_amqp')
        self.config(unicast_address='UNI-CAST', group='oslo_messaging_amqp')
        self.config(anycast_address='ANY-CAST', group='oslo_messaging_amqp')
        self.config(default_notification_exchange='NOTIFY-EXCHANGE', group='oslo_messaging_amqp')
        self.config(default_rpc_exchange='RPC-EXCHANGE', group='oslo_messaging_amqp')
        notifications = [(oslo_messaging.Target(topic='test-topic'), 'info'), (oslo_messaging.Target(topic='test-topic'), 'error'), (oslo_messaging.Target(topic='test-topic'), 'debug')]
        msgs = self._address_test(oslo_messaging.Target(exchange=None, topic='test-topic'), notifications)
        addrs = [m.address for m in msgs]
        notify_addrs = [a for a in addrs if a.startswith('NOTIFY-PREFIX')]
        self.assertEqual(len(notify_addrs), len(notifications))
        self.assertEqual(len(notifications), len([a for a in notify_addrs if 'ANY-CAST' in a]))
        self.assertEqual(len(notifications), len([a for a in notify_addrs if 'NOTIFY-EXCHANGE' in a]))
        rpc_addrs = [a for a in addrs if a.startswith('RPC-PREFIX')]
        self.assertEqual(2, len([a for a in rpc_addrs if 'ANY-CAST' in a]))
        self.assertEqual(1, len([a for a in rpc_addrs if 'MULTI-CAST' in a]))
        self.assertEqual(2, len([a for a in rpc_addrs if 'UNI-CAST' in a]))
        self.assertEqual(len(rpc_addrs), len([a for a in rpc_addrs if 'RPC-EXCHANGE' in a]))

    def _dynamic_test(self, product):
        broker = FakeBroker(self.conf.oslo_messaging_amqp, product=product)
        broker.start()
        url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d' % (broker.host, broker.port))
        driver = amqp_driver.ProtonDriver(self.conf, url)
        target = oslo_messaging.Target(topic='test-topic', server='Server')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
        driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
        listener.join(timeout=30)
        addresser = driver._ctrl.addresser
        driver.cleanup()
        broker.stop()
        return addresser

    def test_dynamic_addressing(self):
        self.config(addressing_mode='dynamic', group='oslo_messaging_amqp')
        self.assertIsInstance(self._dynamic_test('router'), RoutableAddresser)
        self.assertIsInstance(self._dynamic_test('qpid-cpp'), LegacyAddresser)