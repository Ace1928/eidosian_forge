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
class TestAmqpSend(_AmqpBrokerTestCaseAuto):
    """Test sending and receiving messages."""

    def test_driver_unconnected_cleanup(self):
        """Verify the driver can cleanly shutdown even if never connected."""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver.cleanup()

    def test_listener_cleanup(self):
        """Verify unused listener can cleanly shutdown."""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = driver.listen(target, None, None)._poll_style_listener
        self.assertIsInstance(listener, amqp_driver.ProtonListener)
        driver.cleanup()

    def test_send_no_reply(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
        rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
        self.assertIsNone(rc)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        self.assertEqual({'msg': 'value'}, listener.messages.get().message)
        predicate = lambda: self._broker.sender_link_ack_count == 1
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.cleanup()

    def test_send_exchange_with_reply(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target1 = oslo_messaging.Target(topic='test-topic', exchange='e1')
        listener1 = _ListenerThread(driver.listen(target1, None, None)._poll_style_listener, 1)
        target2 = oslo_messaging.Target(topic='test-topic', exchange='e2')
        listener2 = _ListenerThread(driver.listen(target2, None, None)._poll_style_listener, 1)
        rc = driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'e1'}, wait_for_reply=True, timeout=30)
        self.assertIsNotNone(rc)
        self.assertEqual('e1', rc.get('correlation-id'))
        rc = driver.send(target2, {'context': 'whatever'}, {'method': 'echo', 'id': 'e2'}, wait_for_reply=True, timeout=30)
        self.assertIsNotNone(rc)
        self.assertEqual('e2', rc.get('correlation-id'))
        listener1.join(timeout=30)
        self.assertFalse(listener1.is_alive())
        listener2.join(timeout=30)
        self.assertFalse(listener2.is_alive())
        driver.cleanup()

    def test_messaging_patterns(self):
        """Verify the direct, shared, and fanout message patterns work."""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target1 = oslo_messaging.Target(topic='test-topic', server='server1')
        listener1 = _ListenerThread(driver.listen(target1, None, None)._poll_style_listener, 4)
        target2 = oslo_messaging.Target(topic='test-topic', server='server2')
        listener2 = _ListenerThread(driver.listen(target2, None, None)._poll_style_listener, 3)
        shared_target = oslo_messaging.Target(topic='test-topic')
        fanout_target = oslo_messaging.Target(topic='test-topic', fanout=True)
        driver.send(shared_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'either-1'}, wait_for_reply=True)
        self.assertEqual(1, self._broker.topic_count)
        self.assertEqual(1, self._broker.direct_count)
        driver.send(shared_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'either-2'}, wait_for_reply=True)
        self.assertEqual(2, self._broker.topic_count)
        self.assertEqual(2, self._broker.direct_count)
        driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'server1-1'}, wait_for_reply=True)
        driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'server1-2'}, wait_for_reply=True)
        self.assertEqual(6, self._broker.direct_count)
        driver.send(target2, {'context': 'whatever'}, {'method': 'echo', 'id': 'server2'}, wait_for_reply=True)
        self.assertEqual(8, self._broker.direct_count)
        driver.send(fanout_target, {'context': 'whatever'}, {'method': 'echo', 'id': 'fanout'})
        listener1.join(timeout=30)
        self.assertFalse(listener1.is_alive())
        listener2.join(timeout=30)
        self.assertFalse(listener2.is_alive())
        self.assertEqual(1, self._broker.fanout_count)
        listener1_ids = [x.message.get('id') for x in listener1.get_messages()]
        listener2_ids = [x.message.get('id') for x in listener2.get_messages()]
        self.assertTrue('fanout' in listener1_ids and 'fanout' in listener2_ids)
        self.assertTrue('server1-1' in listener1_ids and 'server1-1' not in listener2_ids)
        self.assertTrue('server1-2' in listener1_ids and 'server1-2' not in listener2_ids)
        self.assertTrue('server2' in listener2_ids and 'server2' not in listener1_ids)
        if 'either-1' in listener1_ids:
            self.assertTrue('either-2' in listener2_ids and 'either-2' not in listener1_ids and ('either-1' not in listener2_ids))
        else:
            self.assertTrue('either-2' in listener1_ids and 'either-2' not in listener2_ids and ('either-1' in listener2_ids))
        predicate = lambda: self._broker.sender_link_ack_count == 12
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.cleanup()

    def test_send_timeout(self):
        """Verify send timeout - no reply sent."""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
        self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': 'whatever'}, {'method': 'drop'}, wait_for_reply=True, timeout=1.0)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        driver.cleanup()

    def test_released_send(self):
        """Verify exception thrown if send Nacked."""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='no listener')
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send, target, {'context': 'whatever'}, {'method': 'drop'}, wait_for_reply=True, retry=0, timeout=1.0)
        driver.cleanup()

    def test_send_not_acked(self):
        """Verify exception thrown ack dropped."""
        self.config(pre_settled=[], group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver._default_send_timeout = 2
        target = oslo_messaging.Target(topic='!no-ack!')
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send, target, {'context': 'whatever'}, {'method': 'drop'}, retry=0, wait_for_reply=True)
        driver.cleanup()

    def test_no_ack_cast(self):
        """Verify no exception is thrown if acks are turned off"""
        self.config(pre_settled=['rpc-cast'], group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver._default_send_timeout = 2
        target = oslo_messaging.Target(topic='!no-ack!')
        driver.send(target, {'context': 'whatever'}, {'method': 'drop'}, wait_for_reply=False)
        driver.cleanup()

    def test_call_late_reply(self):
        """What happens if reply arrives after timeout?"""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _SlowResponder(driver.listen(target, None, None)._poll_style_listener, delay=3)
        self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': 'whatever'}, {'method': 'echo', 'id': '???'}, wait_for_reply=True, timeout=1.0)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        predicate = lambda: self._broker.sender_link_ack_count == 1
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.cleanup()

    def test_call_failed_reply(self):
        """Send back an exception generated at the listener"""

        class _FailedResponder(_ListenerThread):

            def __init__(self, listener):
                super(_FailedResponder, self).__init__(listener, 1)

            def run(self):
                self.started.set()
                while not self._done.is_set():
                    for in_msg in self.listener.poll(timeout=0.5):
                        try:
                            raise RuntimeError('Oopsie!')
                        except RuntimeError:
                            in_msg.reply(reply=None, failure=sys.exc_info())
                        self._done.set()
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _FailedResponder(driver.listen(target, None, None)._poll_style_listener)
        self.assertRaises(RuntimeError, driver.send, target, {'context': 'whatever'}, {'method': 'echo'}, wait_for_reply=True, timeout=5.0)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        driver.cleanup()

    def test_call_reply_timeout(self):
        """What happens if the replier times out?"""

        class _TimeoutListener(_ListenerThread):

            def __init__(self, listener):
                super(_TimeoutListener, self).__init__(listener, 1)

            def run(self):
                self.started.set()
                while not self._done.is_set():
                    for in_msg in self.listener.poll(timeout=0.5):
                        in_msg._reply_to = '!no-ack!'
                        in_msg.reply(reply={'correlation-id': in_msg.message.get('id')})
                        self._done.set()
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver._default_reply_timeout = 1
        target = oslo_messaging.Target(topic='test-topic')
        listener = _TimeoutListener(driver.listen(target, None, None)._poll_style_listener)
        self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': 'whatever'}, {'method': 'echo'}, wait_for_reply=True, timeout=3)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        driver.cleanup()

    def test_listener_requeue(self):
        """Emulate Server requeue on listener incoming messages"""
        self.config(pre_settled=[], group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver.require_features(requeue=True)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1, msg_ack=False)
        rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
        self.assertIsNone(rc)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        predicate = lambda: self._broker.sender_link_requeue_count == 1
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.cleanup()

    def test_sender_minimal_credit(self):
        self.config(reply_link_credit=1, rpc_server_credit=1, group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic', server='server')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 4)
        for i in range(4):
            threading.Thread(target=driver.send, args=(target, {'context': 'whatever'}, {'method': 'echo'}), kwargs={'wait_for_reply': True}).start()
        predicate = lambda: self._broker.direct_count == 8
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        listener.join(timeout=30)
        driver.cleanup()

    def test_sender_link_maintenance(self):
        self.config(default_sender_link_timeout=1, group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic-maint')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 3)
        rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
        self.assertIsNone(rc)
        predicate = lambda: self._broker.receiver_link_count == 1
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        self.assertTrue(listener.is_alive())
        self.assertEqual({'msg': 'value'}, listener.messages.get().message)
        predicate = lambda: self._broker.receiver_link_count == 0
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
        self.assertIsNone(rc)
        predicate = lambda: self._broker.receiver_link_count == 1
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        self.assertTrue(listener.is_alive())
        self.assertEqual({'msg': 'value'}, listener.messages.get().message)
        predicate = lambda: self._broker.receiver_link_count == 0
        _wait_until(predicate, 30)
        self.assertTrue(predicate())
        driver.cleanup()

    def test_call_monitor_ok(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _CallMonitor(driver.listen(target, None, None)._poll_style_listener, delay=11, hb_count=100)
        rc = driver.send(target, {'context': True}, {'method': 'echo', 'id': '1'}, wait_for_reply=True, timeout=60, call_monitor_timeout=5)
        self.assertIsNotNone(rc)
        self.assertEqual('1', rc.get('correlation-id'))
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        driver.cleanup()

    def test_call_monitor_bad_no_heartbeat(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _CallMonitor(driver.listen(target, None, None)._poll_style_listener, delay=11, hb_count=1)
        self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': True}, {'method': 'echo', 'id': '1'}, wait_for_reply=True, timeout=60, call_monitor_timeout=5)
        listener.kill()
        self.assertFalse(listener.is_alive())
        driver.cleanup()

    def test_call_monitor_bad_call_timeout(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _CallMonitor(driver.listen(target, None, None)._poll_style_listener, delay=20, hb_count=100)
        self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': True}, {'method': 'echo', 'id': '1'}, wait_for_reply=True, timeout=11, call_monitor_timeout=5)
        listener.kill()
        self.assertFalse(listener.is_alive())
        driver.cleanup()