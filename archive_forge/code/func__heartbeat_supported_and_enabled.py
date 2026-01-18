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
def _heartbeat_supported_and_enabled(self):
    if self.heartbeat_timeout_threshold <= 0:
        return False
    if self.connection.supports_heartbeats:
        return True
    elif not self._heartbeat_support_log_emitted:
        LOG.warning('Heartbeat support requested but it is not supported by the kombu driver or the broker')
        self._heartbeat_support_log_emitted = True
    return False