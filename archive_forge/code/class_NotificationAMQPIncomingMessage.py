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
class NotificationAMQPIncomingMessage(AMQPIncomingMessage):

    def acknowledge(self):

        def _do_ack():
            try:
                self.message.acknowledge()
            except Exception as exc:
                LOG.warning('Failed to acknowledge received message: %s', exc)
        self._message_operations_handler.do(_do_ack)
        self.listener.msg_id_cache.add(self.unique_id)

    def requeue(self):

        def _do_requeue():
            try:
                self.message.requeue()
            except Exception as exc:
                LOG.warning('Failed to requeue received message: %s', exc)
        self._message_operations_handler.do(_do_requeue)