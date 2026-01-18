import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import task
class PrinterTask(task.Task):

    def __init__(self, name, show_name=True, inject=None):
        super(PrinterTask, self).__init__(name, inject=inject)
        self._show_name = show_name

    def execute(self, output):
        if self._show_name:
            print('%s: %s' % (self.name, output))
        else:
            print(output)