from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
class CometTracer(BaseTracer):
    """Comet Tracer."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Comet Tracer."""
        super().__init__(**kwargs)
        self._span_map: Dict['UUID', 'Span'] = {}
        'Map from run id to span.'
        self._chains_map: Dict['UUID', 'Chain'] = {}
        'Map from run id to chain.'
        self._initialize_comet_modules()

    def _initialize_comet_modules(self) -> None:
        comet_llm_api = import_comet_llm_api()
        self._chain: ModuleType = comet_llm_api.chain
        self._span: ModuleType = comet_llm_api.span
        self._chain_api: ModuleType = comet_llm_api.chain_api
        self._experiment_info: ModuleType = comet_llm_api.experiment_info
        self._flush: Callable[[], None] = comet_llm_api.flush

    def _persist_run(self, run: 'Run') -> None:
        chain_ = self._chains_map[run.id]
        chain_.set_outputs(outputs=run.outputs)
        self._chain_api.log_chain(chain_)

    def _process_start_trace(self, run: 'Run') -> None:
        if not run.parent_run_id:
            chain_: 'Chain' = self._chain.Chain(inputs=run.inputs, metadata=None, experiment_info=self._experiment_info.get())
            self._chains_map[run.id] = chain_
        else:
            span: 'Span' = self._span.Span(inputs=run.inputs, category=_get_run_type(run), metadata=run.extra, name=run.name)
            span.__api__start__(self._chains_map[run.parent_run_id])
            self._chains_map[run.id] = self._chains_map[run.parent_run_id]
            self._span_map[run.id] = span

    def _process_end_trace(self, run: 'Run') -> None:
        if not run.parent_run_id:
            pass
        else:
            span = self._span_map[run.id]
            span.set_outputs(outputs=run.outputs)
            span.__api__end__()

    def flush(self) -> None:
        self._flush()

    def _on_llm_start(self, run: 'Run') -> None:
        """Process the LLM Run upon start."""
        self._process_start_trace(run)

    def _on_llm_end(self, run: 'Run') -> None:
        """Process the LLM Run."""
        self._process_end_trace(run)

    def _on_llm_error(self, run: 'Run') -> None:
        """Process the LLM Run upon error."""
        self._process_end_trace(run)

    def _on_chain_start(self, run: 'Run') -> None:
        """Process the Chain Run upon start."""
        self._process_start_trace(run)

    def _on_chain_end(self, run: 'Run') -> None:
        """Process the Chain Run."""
        self._process_end_trace(run)

    def _on_chain_error(self, run: 'Run') -> None:
        """Process the Chain Run upon error."""
        self._process_end_trace(run)

    def _on_tool_start(self, run: 'Run') -> None:
        """Process the Tool Run upon start."""
        self._process_start_trace(run)

    def _on_tool_end(self, run: 'Run') -> None:
        """Process the Tool Run."""
        self._process_end_trace(run)

    def _on_tool_error(self, run: 'Run') -> None:
        """Process the Tool Run upon error."""
        self._process_end_trace(run)