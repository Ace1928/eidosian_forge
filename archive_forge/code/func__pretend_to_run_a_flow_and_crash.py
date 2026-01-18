import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def _pretend_to_run_a_flow_and_crash(self, when):
    flow = uf.Flow('flow-1', retry.Times(3, provides='x')).add(utils.ProgressingTask('task1'))
    engine = self._make_engine(flow)
    engine.compile()
    engine.prepare()
    engine.storage.set_flow_state(st.RUNNING)
    engine.storage.set_atom_intention('flow-1_retry', st.EXECUTE)
    engine.storage.set_atom_intention('task1', st.EXECUTE)
    engine.storage.save('flow-1_retry', 1)
    fail = failure.Failure.from_exception(RuntimeError('foo'))
    engine.storage.save('task1', fail, state=st.FAILURE)
    if when == 'task fails':
        return engine
    engine.storage.save_retry_failure('flow-1_retry', 'task1', fail)
    if when == 'retry queried':
        return engine
    engine.storage.set_atom_intention('flow-1_retry', st.RETRY)
    if when == 'retry updated':
        return engine
    engine.storage.set_atom_intention('task1', st.REVERT)
    if when == 'task updated':
        return engine
    engine.storage.set_atom_state('task1', st.REVERTING)
    if when == 'revert scheduled':
        return engine
    raise ValueError('Invalid crash point: %s' % when)