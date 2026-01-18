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
class AMQPListener(base.PollStyleListener):
    use_cache = False

    def __init__(self, driver, conn):
        super(AMQPListener, self).__init__(driver.prefetch_size)
        self.driver = driver
        self.conn = conn
        self.msg_id_cache = rpc_amqp._MsgIdCache()
        self.incoming = []
        self._shutdown = eventletutils.Event()
        self._shutoff = eventletutils.Event()
        self._obsolete_reply_queues = ObsoleteReplyQueuesCache()
        self._message_operations_handler = MessageOperationsHandler('AMQPListener')
        self._current_timeout = ACK_REQUEUE_EVERY_SECONDS_MIN

    def __call__(self, message):
        ctxt = rpc_amqp.unpack_context(message)
        try:
            unique_id = self.msg_id_cache.check_duplicate_message(message)
        except rpc_common.DuplicateMessageError:
            LOG.exception('ignoring duplicate message %s', ctxt.msg_id)
            return
        if self.use_cache:
            self.msg_id_cache.add(unique_id)
        if ctxt.msg_id:
            LOG.debug('received message msg_id: %(msg_id)s reply to %(queue)s', {'queue': ctxt.reply_q, 'msg_id': ctxt.msg_id})
        else:
            LOG.debug('received message with unique_id: %s', unique_id)
        self.incoming.append(self.message_cls(self, ctxt.to_dict(), message, unique_id, ctxt.msg_id, ctxt.reply_q, ctxt.client_timeout, self._obsolete_reply_queues, self._message_operations_handler))

    @base.batch_poll_helper
    def poll(self, timeout=None):
        stopwatch = timeutils.StopWatch(duration=timeout).start()
        while not self._shutdown.is_set():
            self._message_operations_handler.process()
            LOG.debug('Listener is running')
            if self.incoming:
                LOG.debug('Poll the incoming message with unique_id: %s', self.incoming[0].unique_id)
                return self.incoming.pop(0)
            left = stopwatch.leftover(return_none=True)
            if left is None:
                left = self._current_timeout
            if left <= 0:
                return None
            try:
                LOG.debug('AMQPListener connection consume')
                self.conn.consume(timeout=min(self._current_timeout, left))
            except rpc_common.Timeout:
                LOG.debug('AMQPListener connection timeout')
                self._current_timeout = min(self._current_timeout * 2, ACK_REQUEUE_EVERY_SECONDS_MAX)
            else:
                self._current_timeout = ACK_REQUEUE_EVERY_SECONDS_MIN
        LOG.debug('Listener is stopped')
        self._message_operations_handler.process()
        if self.incoming:
            LOG.debug('Poll the incoming message with unique_id: %s', self.incoming[0].unique_id)
            return self.incoming.pop(0)
        self._shutoff.set()

    def stop(self):
        self._shutdown.set()
        self.conn.stop_consuming()
        self._shutoff.wait()
        self._message_operations_handler.process_in_background()

    def cleanup(self):
        self._message_operations_handler.stop()
        self.conn.close()