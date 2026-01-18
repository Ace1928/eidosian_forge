import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
class Adder(task.Task):

    def execute(self, x, y):
        return x + y