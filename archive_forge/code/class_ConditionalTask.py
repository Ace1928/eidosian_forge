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
class ConditionalTask(ProgressingTask):

    def execute(self, x, y):
        super(ConditionalTask, self).execute()
        if x != y:
            raise RuntimeError('Woot!')