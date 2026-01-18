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
class TestFailover(test_utils.BaseTestCase):

    def setUp(self):
        super(TestFailover, self).setUp()
        self.config(addressing_mode='dynamic', group='oslo_messaging_amqp')
        self._brokers = self._gen_brokers()
        self._primary = 0
        self._backup = 1
        hosts = []
        for broker in self._brokers:
            hosts.append(oslo_messaging.TransportHost(hostname=broker.host, port=broker.port))
        self._broker_url = self._gen_transport_url(hosts)

    def tearDown(self):
        super(TestFailover, self).tearDown()
        for broker in self._brokers:
            if broker.is_alive():
                broker.stop()

    def _gen_brokers(self):
        return [FakeBroker(self.conf.oslo_messaging_amqp, product='qpid-cpp'), FakeBroker(self.conf.oslo_messaging_amqp, product='routable')]

    def _gen_transport_url(self, hosts):
        return oslo_messaging.TransportURL(self.conf, transport='amqp', hosts=hosts)

    def _failover(self, fail_broker):
        self._brokers[0].start()
        self._brokers[1].start()
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='my-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 2)
        predicate = lambda: self._brokers[0].sender_link_count == 4 or self._brokers[1].sender_link_count == 4
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        if self._brokers[1].sender_link_count == 4:
            self._primary = 1
            self._backup = 0
        rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'echo-1'}, wait_for_reply=True, timeout=30)
        self.assertIsNotNone(rc)
        self.assertEqual('echo-1', rc.get('correlation-id'))
        self.assertEqual(1, self._brokers[self._primary].topic_count)
        self.assertEqual(1, self._brokers[self._primary].direct_count)
        fail_broker(self._brokers[self._primary])
        predicate = lambda: self._brokers[self._backup].sender_link_count == 4
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'echo-2'}, wait_for_reply=True, timeout=2)
        self.assertIsNotNone(rc)
        self.assertEqual('echo-2', rc.get('correlation-id'))
        self.assertEqual(1, self._brokers[self._backup].topic_count)
        self.assertEqual(1, self._brokers[self._backup].direct_count)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        self._brokers[self._backup].stop()
        driver.cleanup()

    def test_broker_crash(self):
        """Simulate a failure of one broker."""

        def _meth(broker):
            broker.stop()
            time.sleep(0.5)
        self._failover(_meth)

    def test_broker_shutdown(self):
        """Simulate a normal shutdown of a broker."""

        def _meth(broker):
            broker.stop(clean=True)
            time.sleep(0.5)
        self._failover(_meth)

    def test_heartbeat_failover(self):
        """Simulate broker heartbeat timeout."""

        def _meth(broker):
            broker.pause()
        self.config(idle_timeout=2, group='oslo_messaging_amqp')
        self._failover(_meth)
        self._brokers[self._primary].stop()

    def test_listener_failover(self):
        """Verify that Listeners sharing the same topic are re-established
        after failover.
        """
        self._brokers[0].start()
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='my-topic')
        bcast = oslo_messaging.Target(topic='my-topic', fanout=True)
        listener1 = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 2)
        listener2 = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 2)
        predicate = lambda: self._brokers[0].sender_link_count == 7
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.send(bcast, {'context': 'whatever'}, {'method': 'ignore', 'id': 'echo-1'})
        predicate = lambda: self._brokers[0].fanout_sent_count == 2
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        self._brokers[1].start()
        self._brokers[0].stop(clean=True)
        predicate = lambda: self._brokers[1].sender_link_count == 7
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.send(bcast, {'context': 'whatever'}, {'method': 'ignore', 'id': 'echo-2'})
        predicate = lambda: self._brokers[1].fanout_sent_count == 2
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        listener1.join(timeout=30)
        listener2.join(timeout=30)
        self.assertFalse(listener1.is_alive() or listener2.is_alive())
        driver.cleanup()
        self._brokers[1].stop()