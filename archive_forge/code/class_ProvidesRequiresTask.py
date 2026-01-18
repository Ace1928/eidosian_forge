import contextlib
import string
import threading
import time
from oslo_utils import timeutils
import redis
from taskflow import exceptions
from taskflow.listeners import capturing
from taskflow.persistence.backends import impl_memory
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from taskflow.utils import kazoo_utils
from taskflow.utils import redis_utils
class ProvidesRequiresTask(task.Task):

    def __init__(self, name, provides, requires, return_tuple=True):
        super(ProvidesRequiresTask, self).__init__(name=name, provides=provides, requires=requires)
        self.return_tuple = isinstance(provides, (tuple, list))

    def execute(self, *args, **kwargs):
        if self.return_tuple:
            return tuple(range(len(self.provides)))
        else:
            return dict(((k, k) for k in self.provides))