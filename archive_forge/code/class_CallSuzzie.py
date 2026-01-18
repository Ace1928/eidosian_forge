import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class CallSuzzie(task.Task):

    def execute(self, suzzie_number, *args, **kwargs):
        raise IOError('Suzzie not home right now.')