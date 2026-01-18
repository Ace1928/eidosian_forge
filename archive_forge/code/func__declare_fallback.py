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
def _declare_fallback(self, err, conn, consumer_arguments):
    """Fallback by declaring a non durable queue.

        When a control exchange is shared between services it is possible
        that some service created first a non durable control exchange and
        then after that an other service can try to create the same control
        exchange but as a durable control exchange. In this case RabbitMQ
        will raise an exception (PreconditionFailed), and then it will stop
        our execution and our service will fail entirly. In this case we want
        to fallback by creating a non durable queue to match the default
        config.
        """
    if "PRECONDITION_FAILED - inequivalent arg 'durable'" in str(err):
        LOG.info('[%s] Retrying to declare the exchange (%s) as non durable', conn.connection_id, self.exchange_name)
        self.exchange = kombu.entity.Exchange(name=self.exchange_name, type=self.type, durable=False, auto_delete=self.queue_auto_delete)
        self.queue = kombu.entity.Queue(name=self.queue_name, channel=conn.channel, exchange=self.exchange, durable=False, auto_delete=self.queue_auto_delete, routing_key=self.routing_key, queue_arguments=self.queue_arguments, consumer_arguments=consumer_arguments)
        self.queue.declare()