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
class TestEventTimeListener(test.TestCase, EngineMakerMixin):

    def test_event_time(self):
        flow = lf.Flow('flow1').add(SleepyTask('task1', sleep_for=0.1))
        engine = self._make_engine(flow)
        with timing.EventTimeListener(engine):
            engine.run()
        t_uuid = engine.storage.get_atom_uuid('task1')
        td = engine.storage._flowdetail.find(t_uuid)
        self.assertIsNotNone(td)
        self.assertIsNotNone(td.meta)
        running_field = '%s-timestamp' % states.RUNNING
        success_field = '%s-timestamp' % states.SUCCESS
        self.assertIn(running_field, td.meta)
        self.assertIn(success_field, td.meta)
        td_duration = td.meta[success_field] - td.meta[running_field]
        self.assertGreaterEqual(0.1, td_duration)
        fd_meta = engine.storage._flowdetail.meta
        self.assertIn(running_field, fd_meta)
        self.assertIn(success_field, fd_meta)
        fd_duration = fd_meta[success_field] - fd_meta[running_field]
        self.assertGreaterEqual(0.1, fd_duration)