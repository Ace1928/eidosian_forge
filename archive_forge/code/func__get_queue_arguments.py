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
def _get_queue_arguments(rabbit_ha_queues, rabbit_queue_ttl, rabbit_quorum_queue, rabbit_quorum_queue_config, rabbit_stream_fanout):
    """Construct the arguments for declaring a queue.

    If the rabbit_ha_queues option is set, we try to declare a mirrored queue
    as described here:

      http://www.rabbitmq.com/ha.html

    Setting x-ha-policy to all means that the queue will be mirrored
    to all nodes in the cluster. In RabbitMQ 3.0, queue mirroring is
    no longer controlled by the x-ha-policy argument when declaring a
    queue. If you just want to make sure that all queues (except those
    with auto-generated names) are mirrored across all nodes, run:
      rabbitmqctl set_policy HA '^(?!amq\\.).*' '{"ha-mode": "all"}'

    If the rabbit_queue_ttl option is > 0, then the queue is
    declared with the "Queue TTL" value as described here:

      https://www.rabbitmq.com/ttl.html

    Setting a queue TTL causes the queue to be automatically deleted
    if it is unused for the TTL duration.  This is a helpful safeguard
    to prevent queues with zero consumers from growing without bound.

    If the rabbit_quorum_queue option is set, we try to declare a mirrored
    queue as described here:

      https://www.rabbitmq.com/quorum-queues.html

    Setting x-queue-type to quorum means that replicated FIFO queue based on
    the Raft consensus algorithm will be used. It is available as of
    RabbitMQ 3.8.0. If set this option will conflict with
    the HA queues (``rabbit_ha_queues``) aka mirrored queues,
    in other words HA queues should be disabled.

    rabbit_quorum_queue_config:
    Quorum queues provides three options to handle message poisoning
    and limit the resources the quorum queue can use
    x-delivery-limit number of times the queue will try to deliver
    a message before it decide to discard it
    x-max-in-memory-length, x-max-in-memory-bytes control the size
    of memory used by quorum queue

    If the rabbit_stream_fanout option is set, fanout queues are going to use
    stream instead of quorum queues. See here:
      https://www.rabbitmq.com/streams.html
    """
    args = {}
    if rabbit_quorum_queue and rabbit_ha_queues:
        raise RuntimeError('Configuration Error: rabbit_quorum_queue and rabbit_ha_queues both enabled, queue type is quorum or HA (mirrored) not both')
    if rabbit_ha_queues:
        args['x-ha-policy'] = 'all'
    if rabbit_quorum_queue:
        args['x-queue-type'] = 'quorum'
        if rabbit_quorum_queue_config.delivery_limit:
            args['x-delivery-limit'] = rabbit_quorum_queue_config.delivery_limit
        if rabbit_quorum_queue_config.max_memory_length:
            args['x-max-in-memory-length'] = rabbit_quorum_queue_config.max_memory_length
        if rabbit_quorum_queue_config.max_memory_bytes:
            args['x-max-in-memory-bytes'] = rabbit_quorum_queue_config.max_memory_bytes
    if rabbit_queue_ttl > 0:
        args['x-expires'] = rabbit_queue_ttl * 1000
    if rabbit_stream_fanout:
        args = {'x-queue-type': 'stream'}
        if rabbit_queue_ttl > 0:
            args['x-max-age'] = f'{rabbit_queue_ttl}s'
    return args