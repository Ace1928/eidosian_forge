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
def _transform_transport_url(self, url, host, default_username='', default_password='', default_hostname=''):
    transport = url.transport.replace('kombu+', '')
    transport = transport.replace('rabbit', 'amqp')
    return '%s://%s:%s@%s:%s/%s' % (transport, parse.quote(host.username or default_username), parse.quote(host.password or default_password), self._parse_url_hostname(host.hostname) or default_hostname, str(host.port or 5672), url.virtual_host or '')