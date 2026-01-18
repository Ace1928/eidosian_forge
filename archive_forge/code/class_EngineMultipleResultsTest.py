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
class EngineMultipleResultsTest(utils.EngineTestBase):

    def test_fetch_with_a_single_result(self):
        flow = lf.Flow('flow')
        flow.add(utils.TaskOneReturn(provides='x'))
        engine = self._make_engine(flow)
        engine.run()
        result = engine.storage.fetch('x')
        self.assertEqual(1, result)

    def test_many_results_visible_to(self):
        flow = lf.Flow('flow')
        flow.add(utils.AddOneSameProvidesRequires('a', rebind={'value': 'source'}))
        flow.add(utils.AddOneSameProvidesRequires('b'))
        flow.add(utils.AddOneSameProvidesRequires('c'))
        engine = self._make_engine(flow, store={'source': 0})
        engine.run()
        atoms = list(flow)
        a = atoms[0]
        a_kwargs = engine.storage.fetch_mapped_args(a.rebind, atom_name='a')
        self.assertEqual({'value': 0}, a_kwargs)
        b = atoms[1]
        b_kwargs = engine.storage.fetch_mapped_args(b.rebind, atom_name='b')
        self.assertEqual({'value': 1}, b_kwargs)
        c = atoms[2]
        c_kwargs = engine.storage.fetch_mapped_args(c.rebind, atom_name='c')
        self.assertEqual({'value': 2}, c_kwargs)

    def test_many_results_storage_provided_visible_to(self):
        flow = lf.Flow('flow')
        flow.add(utils.AddOneSameProvidesRequires('a'))
        flow.add(utils.AddOneSameProvidesRequires('b'))
        flow.add(utils.AddOneSameProvidesRequires('c'))
        engine = self._make_engine(flow, store={'value': 0})
        engine.run()
        atoms = list(flow)
        a = atoms[0]
        a_kwargs = engine.storage.fetch_mapped_args(a.rebind, atom_name='a')
        self.assertEqual({'value': 0}, a_kwargs)
        b = atoms[1]
        b_kwargs = engine.storage.fetch_mapped_args(b.rebind, atom_name='b')
        self.assertEqual({'value': 0}, b_kwargs)
        c = atoms[2]
        c_kwargs = engine.storage.fetch_mapped_args(c.rebind, atom_name='c')
        self.assertEqual({'value': 0}, c_kwargs)

    def test_fetch_with_two_results(self):
        flow = lf.Flow('flow')
        flow.add(utils.TaskOneReturn(provides='x'))
        engine = self._make_engine(flow, store={'x': 0})
        engine.run()
        result = engine.storage.fetch('x')
        self.assertEqual(0, result)

    def test_fetch_all_with_a_single_result(self):
        flow = lf.Flow('flow')
        flow.add(utils.TaskOneReturn(provides='x'))
        engine = self._make_engine(flow)
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'x': 1}, result)

    def test_fetch_all_with_two_results(self):
        flow = lf.Flow('flow')
        flow.add(utils.TaskOneReturn(provides='x'))
        engine = self._make_engine(flow, store={'x': 0})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'x': [0, 1]}, result)

    def test_task_can_update_value(self):
        flow = lf.Flow('flow')
        flow.add(utils.TaskOneArgOneReturn(requires='x', provides='x'))
        engine = self._make_engine(flow, store={'x': 0})
        engine.run()
        result = engine.storage.fetch_all()
        self.assertEqual({'x': [0, 1]}, result)