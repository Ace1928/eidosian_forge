import logging
import os
import sys
import time
import taskflow.engines
from taskflow import exceptions
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.tests import utils
from taskflow.types import failure
import example_utils as eu  # noqa
class FirstTask(task.Task):

    def execute(self, sleep1, raise1):
        time.sleep(sleep1)
        if not isinstance(raise1, bool):
            raise TypeError('Bad raise1 value: %r' % raise1)
        if raise1:
            raise FirstException('First task failed')