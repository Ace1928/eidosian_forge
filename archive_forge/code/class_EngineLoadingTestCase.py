import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
class EngineLoadingTestCase(test.TestCase):

    def _make_dummy_flow(self):
        f = linear_flow.Flow('test')
        f.add(test_utils.TaskOneReturn('run-1'))
        return f

    def test_default_load(self):
        f = self._make_dummy_flow()
        e = taskflow.engines.load(f)
        self.assertIsNotNone(e)

    def test_unknown_load(self):
        f = self._make_dummy_flow()
        self.assertRaises(exc.NotFound, taskflow.engines.load, f, engine='not_really_any_engine')

    def test_options_empty(self):
        f = self._make_dummy_flow()
        e = taskflow.engines.load(f)
        self.assertEqual({}, e.options)

    def test_options_passthrough(self):
        f = self._make_dummy_flow()
        e = taskflow.engines.load(f, pass_1=1, pass_2=2)
        self.assertEqual({'pass_1': 1, 'pass_2': 2}, e.options)