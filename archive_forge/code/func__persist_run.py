from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _persist_run(self, run: 'Run') -> None:
    chain_ = self._chains_map[run.id]
    chain_.set_outputs(outputs=run.outputs)
    self._chain_api.log_chain(chain_)