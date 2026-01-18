from automaton import exceptions as excp
from automaton import runners
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import runtime
from taskflow.patterns import linear_flow as lf
from taskflow import states as st
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import notifier
from taskflow.utils import persistence_utils as pu
def _make_runtime(self, flow, initial_state=None):
    compilation = compiler.PatternCompiler(flow).compile()
    flow_detail = pu.create_flow_detail(flow)
    store = storage.Storage(flow_detail)
    nodes_iter = compilation.execution_graph.nodes(data=True)
    for node, node_attrs in nodes_iter:
        if node_attrs['kind'] in ('task', 'retry'):
            store.ensure_atom(node)
    if initial_state:
        store.set_flow_state(initial_state)
    atom_notifier = notifier.Notifier()
    task_executor = executor.SerialTaskExecutor()
    retry_executor = executor.SerialRetryExecutor()
    task_executor.start()
    self.addCleanup(task_executor.stop)
    r = runtime.Runtime(compilation, store, atom_notifier, task_executor, retry_executor)
    r.compile()
    return r