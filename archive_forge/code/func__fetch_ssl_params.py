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
def _fetch_ssl_params(self):
    """Handles fetching what ssl params should be used for the connection
        (if any).
        """
    if self.ssl:
        ssl_params = dict()
        if self.ssl_version:
            ssl_params['ssl_version'] = self.validate_ssl_version(self.ssl_version)
        if self.ssl_key_file:
            ssl_params['keyfile'] = self.ssl_key_file
        if self.ssl_cert_file:
            ssl_params['certfile'] = self.ssl_cert_file
        if self.ssl_ca_file:
            ssl_params['ca_certs'] = self.ssl_ca_file
            ssl_params['cert_reqs'] = ssl.CERT_REQUIRED
        return ssl_params or True
    return False