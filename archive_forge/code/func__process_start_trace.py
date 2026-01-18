from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _process_start_trace(self, run: 'Run') -> None:
    if not run.parent_run_id:
        chain_: 'Chain' = self._chain.Chain(inputs=run.inputs, metadata=None, experiment_info=self._experiment_info.get())
        self._chains_map[run.id] = chain_
    else:
        span: 'Span' = self._span.Span(inputs=run.inputs, category=_get_run_type(run), metadata=run.extra, name=run.name)
        span.__api__start__(self._chains_map[run.parent_run_id])
        self._chains_map[run.id] = self._chains_map[run.parent_run_id]
        self._span_map[run.id] = span