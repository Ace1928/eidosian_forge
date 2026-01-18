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
def _get_reply_q(self):
    with self._reply_q_lock:
        if self._reply_q is not None:
            return self._reply_q
        if self._q_manager:
            reply_q = 'reply_' + self._q_manager.get()
        else:
            reply_q = 'reply_' + uuid.uuid4().hex
        LOG.info('Creating reply queue: %s', reply_q)
        conn = self._get_connection(rpc_common.PURPOSE_LISTEN)
        self._waiter = ReplyWaiter(reply_q, conn, self._allowed_remote_exmods)
        self._reply_q = reply_q
        self._reply_q_conn = conn
    return self._reply_q