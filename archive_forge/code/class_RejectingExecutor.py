import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
class RejectingExecutor(futurist.GreenThreadPoolExecutor):
    MAX_REJECTIONS_COUNT = 2

    def _reject(self, *args):
        if self._rejections_count < self.MAX_REJECTIONS_COUNT:
            self._rejections_count += 1
            raise futurist.RejectedSubmission()

    def __init__(self):
        self._rejections_count = 0
        super(RejectingExecutor, self).__init__(check_and_reject=self._reject)