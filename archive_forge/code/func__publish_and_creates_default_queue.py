import collections
import contextlib
import errno
import functools
import itertools
import math
import os
import random
import socket
import ssl
import sys
import threading
import time
from urllib import parse
import uuid
from amqp import exceptions as amqp_ex
import kombu
import kombu.connection
import kombu.entity
import kombu.messaging
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging._drivers import pool
from oslo_messaging import _utils
from oslo_messaging import exceptions
def _publish_and_creates_default_queue(self, exchange, msg, routing_key=None, timeout=None, transport_options=None):
    """Publisher that declares a default queue

        When the exchange is missing instead of silently creates an exchange
        not binded to a queue, this publisher creates a default queue
        named with the routing_key

        This is mainly used to not miss notification in case of nobody consumes
        them yet. If the future consumer bind the default queue it can retrieve
        missing messages.

        _set_current_channel is responsible to cleanup the cache.
        """
    queue_identifier = (exchange.name, routing_key)
    if queue_identifier not in self._declared_queues:
        queue = kombu.entity.Queue(channel=self.channel, exchange=exchange, durable=exchange.durable, auto_delete=exchange.auto_delete, name=routing_key, routing_key=routing_key, queue_arguments=_get_queue_arguments(self.rabbit_ha_queues, 0, self.rabbit_quorum_queue, self.rabbit_quorum_queue_config, False))
        log_info = {'key': routing_key, 'exchange': exchange}
        LOG.trace('Connection._publish_and_creates_default_queue: declare queue %(key)s on %(exchange)s exchange', log_info)
        queue.declare()
        self._declared_queues.add(queue_identifier)
    self._publish(exchange, msg, routing_key=routing_key, timeout=timeout)