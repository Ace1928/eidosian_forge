from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _on_llm_error(self, run: 'Run') -> None:
    """Process the LLM Run upon error."""
    self._process_end_trace(run)