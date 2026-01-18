import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def _timeout_task_without_return_but_with_done(self):
    if self.num_runs == 0:
        raise loopingcall.LoopingCallDone(False)
    else:
        self.num_runs = self.num_runs - 1