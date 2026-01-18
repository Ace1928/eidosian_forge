from __future__ import absolute_import, print_function, division
import abc
import logging
import sys
import time
from petl.compat import PY3
from petl.util.base import Table
from petl.util.statistics import onlinestats
class ProgressView(ProgressViewBase):
    """
    Reports progress to a file_object like sys.stdout or a file handler
    """

    def __init__(self, inner, batchsize, prefix, out):
        if out is None:
            self.file_object = sys.stderr
        else:
            self.file_object = out
        super(ProgressView, self).__init__(inner, batchsize, prefix)

    def print_message(self, message):
        print(message, file=self.file_object)
        if hasattr(self.file_object, 'flush'):
            self.file_object.flush()