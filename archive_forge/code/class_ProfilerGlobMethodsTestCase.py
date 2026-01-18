import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class ProfilerGlobMethodsTestCase(test.TestCase):

    def test_get_profiler_not_inited(self):
        profiler.clean()
        self.assertIsNone(profiler.get())

    def test_get_profiler_and_init(self):
        p = profiler.init('secret', base_id='1', parent_id='2')
        self.assertEqual(profiler.get(), p)
        self.assertEqual(p.get_base_id(), '1')
        self.assertEqual(p.get_id(), '2')

    def test_start_not_inited(self):
        profiler.clean()
        profiler.start('name')

    def test_start(self):
        p = profiler.init('secret', base_id='1', parent_id='2')
        p.start = mock.MagicMock()
        profiler.start('name', info='info')
        p.start.assert_called_once_with('name', info='info')

    def test_stop_not_inited(self):
        profiler.clean()
        profiler.stop()

    def test_stop(self):
        p = profiler.init('secret', base_id='1', parent_id='2')
        p.stop = mock.MagicMock()
        profiler.stop(info='info')
        p.stop.assert_called_once_with(info='info')