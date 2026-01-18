import collections
import contextlib
import functools
import threading
import futurist
import testtools
import taskflow.engines
from taskflow.engines.action_engine import engine as eng
from taskflow.engines.worker_based import engine as w_eng
from taskflow.engines.worker_based import worker as wkr
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow.persistence import models
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.types import graph as gr
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils as tu
class EngineOptionalRequirementsTest(utils.EngineTestBase):

    def test_expected_optional_multiplers(self):
        flow_no_inject = lf.Flow('flow')
        flow_no_inject.add(utils.OptionalTask(provides='result'))
        flow_inject_a = lf.Flow('flow')
        flow_inject_a.add(utils.OptionalTask(provides='result', inject={'a': 10}))
        flow_inject_b = lf.Flow('flow')
        flow_inject_b.add(utils.OptionalTask(provides='result', inject={'b': 1000}))
        engine = self._make_engine(flow_no_inject, store={'a': 3})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'result': 15}, result)
        engine = self._make_engine(flow_no_inject, store={'a': 3, 'b': 7})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'b': 7, 'result': 21}, result)
        engine = self._make_engine(flow_inject_a, store={'a': 3})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'result': 50}, result)
        engine = self._make_engine(flow_inject_a, store={'a': 3, 'b': 7})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'b': 7, 'result': 70}, result)
        engine = self._make_engine(flow_inject_b, store={'a': 3})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'result': 3000}, result)
        engine = self._make_engine(flow_inject_b, store={'a': 3, 'b': 7})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'a': 3, 'b': 7, 'result': 3000}, result)