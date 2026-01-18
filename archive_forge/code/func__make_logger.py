import contextlib
import logging
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import reflection
from zake import fake_client
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.jobs import backends as jobs
from taskflow.listeners import claims
from taskflow.listeners import logging as logging_listeners
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils
def _make_logger(self, level=logging.DEBUG):
    log = logging.getLogger(reflection.get_callable_name(self._get_test_method()))
    log.propagate = False
    for handler in reversed(log.handlers):
        log.removeHandler(handler)
    handler = test.CapturingLoggingHandler(level=level)
    log.addHandler(handler)
    log.setLevel(level)
    self.addCleanup(handler.reset)
    self.addCleanup(log.removeHandler, handler)
    return (log, handler)