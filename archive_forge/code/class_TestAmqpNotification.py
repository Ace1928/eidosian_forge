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
class TestAmqpNotification(_AmqpBrokerTestCaseAuto):
    """Test sending and receiving notifications."""

    def test_notification(self):
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        notifications = [(oslo_messaging.Target(topic='topic-1'), 'info'), (oslo_messaging.Target(topic='topic-1'), 'error'), (oslo_messaging.Target(topic='topic-2'), 'debug')]
        nl = driver.listen_for_notifications(notifications, None, None, None)._poll_style_listener
        msg_count = len(notifications) * 2
        listener = _ListenerThread(nl, msg_count)
        targets = ['topic-1.info', 'topic-1.bad', 'bad-topic.debug', 'topic-1.error', 'topic-2.debug']
        excepted_targets = []
        for version in (1.0, 2.0):
            for t in targets:
                try:
                    driver.send_notification(oslo_messaging.Target(topic=t), 'context', {'target': t}, version, retry=0)
                except oslo_messaging.MessageDeliveryFailure:
                    excepted_targets.append(t)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
        topics = [x.message.get('target') for x in listener.get_messages()]
        self.assertEqual(msg_count, len(topics))
        self.assertEqual(2, topics.count('topic-1.info'))
        self.assertEqual(2, topics.count('topic-1.error'))
        self.assertEqual(2, topics.count('topic-2.debug'))
        self.assertEqual(4, self._broker.dropped_count)
        self.assertEqual(2, excepted_targets.count('topic-1.bad'))
        self.assertEqual(2, excepted_targets.count('bad-topic.debug'))
        driver.cleanup()

    def test_released_notification(self):
        """Broker sends a Nack (released)"""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send_notification, oslo_messaging.Target(topic='bad address'), 'context', {'target': 'bad address'}, 2.0, retry=0)
        driver.cleanup()

    def test_notification_not_acked(self):
        """Simulate drop of ack from broker"""
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver._default_notify_timeout = 2
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send_notification, oslo_messaging.Target(topic='!no-ack!'), 'context', {'target': '!no-ack!'}, 2.0, retry=0)
        driver.cleanup()

    def test_no_ack_notification(self):
        """Verify no exception is thrown if acks are turned off"""
        self.config(pre_settled=['notify', 'fleabag', 'poochie'], group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        driver._default_notify_timeout = 2
        driver.send_notification(oslo_messaging.Target(topic='!no-ack!'), 'context', {'target': '!no-ack!'}, 2.0)
        driver.cleanup()