import logging
import os
import sys
from concurrent import futures
import taskflow.engines
from taskflow.listeners import base
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import task
from taskflow.types import notifier
class Hi(task.Task):

    def execute(self):
        return 'hi'