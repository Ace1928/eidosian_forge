import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def _raise_and_then_done(self):
    if self.num_runs == 0:
        raise loopingcall.LoopingCallDone(False)
    else:
        self.num_runs = self.num_runs - 1
        raise RuntimeError()