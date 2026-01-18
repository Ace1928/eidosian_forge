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
def _ensure_publishing(self, method, exchange, msg, routing_key=None, timeout=None, retry=None, transport_options=None):
    """Send to a publisher based on the publisher class."""

    def _error_callback(exc):
        log_info = {'topic': exchange.name, 'err_str': exc}
        LOG.error("Failed to publish message to topic '%(topic)s': %(err_str)s", log_info)
        LOG.debug('Exception', exc_info=exc)
    method = functools.partial(method, exchange, msg, routing_key, timeout, transport_options)
    with self._connection_lock:
        self.ensure(method, retry=retry, error_callback=_error_callback)