import logging
import os
import random
import sys
import time
from taskflow import engines
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow import task
class VariableTask(task.Task):

    def __init__(self, name):
        super(VariableTask, self).__init__(name)
        self._sleepy_time = random.random()

    def execute(self):
        time.sleep(self._sleepy_time)