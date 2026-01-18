import contextlib
import functools
import logging
import os
import sys
import time
import traceback
from kazoo import client
from taskflow.conductors import backends as conductor_backends
from taskflow import engines
from taskflow.jobs import backends as job_backends
from taskflow import logging as taskflow_logging
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends as persistence_backends
from taskflow.persistence import models
from taskflow import task
from oslo_utils import timeutils
from oslo_utils import uuidutils
class TakeABottleDown(task.Task):

    def execute(self, bottles_left):
        sys.stdout.write('Take one down, ')
        sys.stdout.flush()
        time.sleep(TAKE_DOWN_DELAY)
        return bottles_left - 1