import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class PrintTask(task.Task):

    def execute(self):
        print("Running '%s'" % self.name)