import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
class TopicWorker(object):
    """A (read-only) worker and its relevant information + useful methods."""
    _NO_IDENTITY = object()

    def __init__(self, topic, tasks, identity=_NO_IDENTITY):
        self.tasks = []
        for task in tasks:
            if not isinstance(task, str):
                task = reflection.get_class_name(task)
            self.tasks.append(task)
        self.topic = topic
        self.identity = identity
        self.last_seen = None

    def performs(self, task):
        if not isinstance(task, str):
            task = reflection.get_class_name(task)
        return task in self.tasks

    def __eq__(self, other):
        if not isinstance(other, TopicWorker):
            return NotImplemented
        if len(other.tasks) != len(self.tasks):
            return False
        if other.topic != self.topic:
            return False
        for task in other.tasks:
            if not self.performs(task):
                return False
        if self._NO_IDENTITY in (self.identity, other.identity):
            return True
        else:
            return other.identity == self.identity

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        r = reflection.get_class_name(self, fully_qualified=False)
        if self.identity is not self._NO_IDENTITY:
            r += '(identity=%s, tasks=%s, topic=%s)' % (self.identity, self.tasks, self.topic)
        else:
            r += '(identity=*, tasks=%s, topic=%s)' % (self.tasks, self.topic)
        return r