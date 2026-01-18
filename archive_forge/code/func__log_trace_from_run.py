from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _log_trace_from_run(self, run: Run) -> None:
    """Logs a LangChain Run to W*B as a W&B Trace."""
    self._ensure_run()
    root_span = self.run_processor.process_span(run)
    model_dict = self.run_processor.process_model(run)
    if root_span is None:
        return
    model_trace = self._trace_tree.WBTraceTree(root_span=root_span, model_dict=model_dict)
    if self._wandb.run is not None:
        self._wandb.run.log({'langchain_trace': model_trace})