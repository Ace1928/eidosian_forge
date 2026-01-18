import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
class FakeExchange(object):

    def __init__(self, name):
        self.name = name
        self._queues_lock = threading.RLock()
        self._topic_queues = {}
        self._server_queues = {}

    def ensure_queue(self, target, pool):
        with self._queues_lock:
            if target.server:
                self._get_server_queue(target.topic, target.server)
            else:
                self._get_topic_queue(target.topic, pool)

    def _get_topic_queue(self, topic, pool=None):
        if pool and (topic, pool) not in self._topic_queues:
            self._topic_queues[topic, pool] = copy.deepcopy(self._get_topic_queue(topic))
        return self._topic_queues.setdefault((topic, pool), [])

    def _get_server_queue(self, topic, server):
        return self._server_queues.setdefault((topic, server), [])

    def deliver_message(self, topic, ctxt, message, server=None, fanout=False, reply_q=None):
        with self._queues_lock:
            if fanout:
                queues = [q for t, q in self._server_queues.items() if t[0] == topic]
            elif server is not None:
                queues = [self._get_server_queue(topic, server)]
            else:
                self._get_topic_queue(topic)
                queues = [q for t, q in self._topic_queues.items() if t[0] == topic]

            def requeue():
                self.deliver_message(topic, ctxt, message, server=server, fanout=fanout, reply_q=reply_q)
            for q in queues:
                q.append((ctxt, message, reply_q, requeue))

    def poll(self, target, pool):
        with self._queues_lock:
            if target.server:
                queue = self._get_server_queue(target.topic, target.server)
            else:
                queue = self._get_topic_queue(target.topic, pool)
            return queue.pop(0) if queue else (None, None, None, None)