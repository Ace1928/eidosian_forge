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
def _publish_and_raises_on_missing_exchange(self, exchange, msg, routing_key=None, timeout=None, transport_options=None):
    """Publisher that raises exception if exchange is missing."""
    if not exchange.passive:
        raise RuntimeError('_publish_and_retry_on_missing_exchange() must be called with an passive exchange.')
    try:
        self._publish(exchange, msg, routing_key=routing_key, timeout=timeout, transport_options=transport_options)
        return
    except self.connection.channel_errors as exc:
        if exc.code == 404:
            raise rpc_amqp.AMQPDestinationNotFound("exchange %s doesn't exist" % exchange.name)
        raise