import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def _run_forever_until_set():
    if wait_ev.is_set():
        raise loopingcall.LoopingCallDone(True)
    else:
        return 0.01