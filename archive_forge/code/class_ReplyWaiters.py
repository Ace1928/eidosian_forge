import logging
import os
import queue
import threading
import time
import uuid
import cachetools
from oslo_concurrency import lockutils
from oslo_utils import eventletutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging import MessageDeliveryFailure
class ReplyWaiters(object):

    def __init__(self):
        self._queues = {}
        self._wrn_threshold = 10

    def get(self, msg_id, timeout):
        try:
            return self._queues[msg_id].get(block=True, timeout=timeout)
        except queue.Empty:
            raise oslo_messaging.MessagingTimeout('Timed out waiting for a reply to message ID %s' % msg_id)

    def put(self, msg_id, message_data):
        LOG.info('Received RPC response for msg %s', msg_id)
        queue = self._queues.get(msg_id)
        if not queue:
            LOG.info('No calling threads waiting for msg_id : %s', msg_id)
            LOG.debug(' queues: %(queues)s, message: %(message)s', {'queues': len(self._queues), 'message': message_data})
        else:
            queue.put(message_data)

    def add(self, msg_id):
        self._queues[msg_id] = queue.Queue()
        queues_length = len(self._queues)
        if queues_length > self._wrn_threshold:
            LOG.warning('Number of call queues is %(queues_length)s, greater than warning threshold: %(old_threshold)s. There could be a leak. Increasing threshold to: %(threshold)s', {'queues_length': queues_length, 'old_threshold': self._wrn_threshold, 'threshold': self._wrn_threshold * 2})
            self._wrn_threshold *= 2

    def remove(self, msg_id):
        del self._queues[msg_id]