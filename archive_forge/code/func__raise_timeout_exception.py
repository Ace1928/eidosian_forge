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
@staticmethod
def _raise_timeout_exception(msg_id, reply_q):
    raise oslo_messaging.MessagingTimeout('Timed out waiting for a reply %(reply_q)s to message ID %(msg_id)s.', {'msg_id': msg_id, 'reply_q': reply_q})