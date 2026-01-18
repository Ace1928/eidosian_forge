from .. import Memory
from ...pipeline.engine.tests.test_engine import EngineTestInterface
from ... import config
class SideEffectInterface(EngineTestInterface):

    def _run_interface(self, runtime):
        global nb_runs
        nb_runs += 1
        return super(SideEffectInterface, self)._run_interface(runtime)