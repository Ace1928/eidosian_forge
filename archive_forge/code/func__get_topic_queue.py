import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
def _get_topic_queue(self, topic, pool=None):
    if pool and (topic, pool) not in self._topic_queues:
        self._topic_queues[topic, pool] = copy.deepcopy(self._get_topic_queue(topic))
    return self._topic_queues.setdefault((topic, pool), [])