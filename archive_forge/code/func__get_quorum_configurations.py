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
def _get_quorum_configurations(self, driver_conf):
    """Get the quorum queue configurations"""
    delivery_limit = driver_conf.rabbit_quorum_delivery_limit
    max_memory_length = driver_conf.rabbit_quorum_max_memory_length
    max_memory_bytes = driver_conf.rabbit_quorum_max_memory_bytes
    return QuorumMemConfig(delivery_limit, max_memory_length, max_memory_bytes)