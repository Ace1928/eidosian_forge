import contextlib
import hashlib
import logging
import os
import random
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils  # noqa
class PrintText(task.Task):

    def __init__(self, print_what, no_slow=False):
        content_hash = hashlib.md5(print_what.encode('utf-8')).hexdigest()[0:8]
        super(PrintText, self).__init__(name='Print: %s' % content_hash)
        self._text = print_what
        self._no_slow = no_slow

    def execute(self):
        if self._no_slow:
            print('-' * len(self._text))
            print(self._text)
            print('-' * len(self._text))
        else:
            with slow_down():
                print('-' * len(self._text))
                print(self._text)
                print('-' * len(self._text))