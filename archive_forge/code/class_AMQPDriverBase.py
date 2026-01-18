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
class AMQPDriverBase(base.BaseDriver):
    missing_destination_retry_timeout = 0

    def __init__(self, conf, url, connection_pool, default_exchange=None, allowed_remote_exmods=None):
        super(AMQPDriverBase, self).__init__(conf, url, default_exchange, allowed_remote_exmods)
        self._default_exchange = default_exchange
        self._connection_pool = connection_pool
        self._reply_q_lock = threading.Lock()
        self._reply_q = None
        self._reply_q_conn = None
        self._waiter = None
        if conf.oslo_messaging_rabbit.use_queue_manager:
            self._q_manager = QManager(hostname=conf.oslo_messaging_rabbit.hostname, processname=conf.oslo_messaging_rabbit.processname)
        else:
            self._q_manager = None

    def _get_exchange(self, target):
        return target.exchange or self._default_exchange

    def _get_connection(self, purpose=rpc_common.PURPOSE_SEND, retry=None):
        return rpc_common.ConnectionContext(self._connection_pool, purpose=purpose, retry=retry)

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

    def _send(self, target, ctxt, message, wait_for_reply=None, timeout=None, call_monitor_timeout=None, envelope=True, notify=False, retry=None, transport_options=None):
        msg = message
        reply_q = None
        if 'method' in msg:
            LOG.debug('Calling RPC method %s on target %s', msg.get('method'), target.topic)
        else:
            LOG.debug('Sending message to topic %s', target.topic)
        if wait_for_reply:
            reply_q = self._get_reply_q()
            msg_id = uuid.uuid4().hex
            msg.update({'_msg_id': msg_id})
            msg.update({'_reply_q': reply_q})
            msg.update({'_timeout': call_monitor_timeout})
            LOG.info('Expecting reply to msg %s in queue %s', msg_id, reply_q)
        rpc_amqp._add_unique_id(msg)
        unique_id = msg[rpc_amqp.UNIQUE_ID]
        rpc_amqp.pack_context(msg, ctxt)
        if envelope:
            msg = rpc_common.serialize_msg(msg)
        if wait_for_reply:
            self._waiter.listen(msg_id)
            log_msg = 'CALL msg_id: %s ' % msg_id
        else:
            log_msg = 'CAST unique_id: %s ' % unique_id
        try:
            with self._get_connection(rpc_common.PURPOSE_SEND, retry) as conn:
                if notify:
                    exchange = self._get_exchange(target)
                    LOG.debug(log_msg + "NOTIFY exchange '%(exchange)s' topic '%(topic)s'", {'exchange': exchange, 'topic': target.topic})
                    conn.notify_send(exchange, target.topic, msg, retry=retry)
                elif target.fanout:
                    log_msg += "FANOUT topic '%(topic)s'" % {'topic': target.topic}
                    LOG.debug(log_msg)
                    conn.fanout_send(target.topic, msg, retry=retry)
                else:
                    topic = target.topic
                    exchange = self._get_exchange(target)
                    if target.server:
                        topic = '%s.%s' % (target.topic, target.server)
                    LOG.debug(log_msg + "exchange '%(exchange)s' topic '%(topic)s'", {'exchange': exchange, 'topic': topic})
                    conn.topic_send(exchange_name=exchange, topic=topic, msg=msg, timeout=timeout, retry=retry, transport_options=transport_options)
            if wait_for_reply:
                result = self._waiter.wait(msg_id, timeout, call_monitor_timeout, reply_q)
                if isinstance(result, Exception):
                    raise result
                return result
        finally:
            if wait_for_reply:
                self._waiter.unlisten(msg_id)

    def send(self, target, ctxt, message, wait_for_reply=None, timeout=None, call_monitor_timeout=None, retry=None, transport_options=None):
        return self._send(target, ctxt, message, wait_for_reply, timeout, call_monitor_timeout, retry=retry, transport_options=transport_options)

    def send_notification(self, target, ctxt, message, version, retry=None):
        return self._send(target, ctxt, message, envelope=version == 2.0, notify=True, retry=retry)

    def listen(self, target, batch_size, batch_timeout):
        conn = self._get_connection(rpc_common.PURPOSE_LISTEN)
        listener = RpcAMQPListener(self, conn)
        conn.declare_topic_consumer(exchange_name=self._get_exchange(target), topic=target.topic, callback=listener)
        conn.declare_topic_consumer(exchange_name=self._get_exchange(target), topic='%s.%s' % (target.topic, target.server), callback=listener)
        conn.declare_fanout_consumer(target.topic, listener)
        return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)

    def listen_for_notifications(self, targets_and_priorities, pool, batch_size, batch_timeout):
        conn = self._get_connection(rpc_common.PURPOSE_LISTEN)
        listener = NotificationAMQPListener(self, conn)
        for target, priority in targets_and_priorities:
            conn.declare_topic_consumer(exchange_name=self._get_exchange(target), topic='%s.%s' % (target.topic, priority), callback=listener, queue_name=pool)
        return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)

    def cleanup(self):
        if self._connection_pool:
            self._connection_pool.empty()
        self._connection_pool = None
        with self._reply_q_lock:
            if self._reply_q is not None:
                self._waiter.stop()
                self._reply_q_conn.close()
                self._reply_q_conn = None
                self._reply_q = None
                self._waiter = None