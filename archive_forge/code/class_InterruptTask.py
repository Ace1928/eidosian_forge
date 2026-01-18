import contextlib
import logging
import os
import sys
from oslo_utils import uuidutils
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
class InterruptTask(task.Task):

    def execute(self):
        engine.suspend()