import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def _next_worker(self, topic, tasks, temporary=False):
    if not temporary:
        w = TopicWorker(topic, tasks, identity=self._seen_workers)
        self._seen_workers += 1
        return w
    else:
        return TopicWorker(topic, tasks)