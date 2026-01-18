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
def _send_reply(self, conn, reply=None, failure=None, ending=True):
    if not self._obsolete_reply_queues.reply_q_valid(self.reply_q, self.msg_id):
        return
    if failure:
        failure = rpc_common.serialize_remote_exception(failure)
    msg = {'result': reply, 'failure': failure, 'ending': ending, '_msg_id': self.msg_id}
    rpc_amqp._add_unique_id(msg)
    unique_id = msg[rpc_amqp.UNIQUE_ID]
    LOG.debug('sending reply msg_id: %(msg_id)s reply queue: %(reply_q)s time elapsed: %(elapsed)ss', {'msg_id': self.msg_id, 'unique_id': unique_id, 'reply_q': self.reply_q, 'elapsed': self.stopwatch.elapsed()})
    conn.direct_send(self.reply_q, rpc_common.serialize_msg(msg))