import logging
import os
import sys
import tempfile
import traceback
from taskflow import engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
class HiTask(task.Task):

    def execute(self):
        print('Hi!')

    def revert(self, **kwargs):
        print('Whooops, said hi too early, take that back!')