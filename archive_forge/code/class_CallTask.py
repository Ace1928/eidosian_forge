import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class CallTask(task.Task):
    """Task that calls person by number."""

    def execute(self, person, number):
        print('Calling %s %s.' % (person, number))