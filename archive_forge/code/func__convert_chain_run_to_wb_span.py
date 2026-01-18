from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _convert_chain_run_to_wb_span(self, run: Run) -> 'Span':
    """Converts a LangChain Chain Run into a W&B Trace Span.
        :param run: The LangChain Chain Run to convert.
        :return: The converted W&B Trace Span.
        """
    base_span = self._convert_run_to_wb_span(run)
    base_span.results = [self.trace_tree.Result(inputs=_serialize_io(run.inputs), outputs=_serialize_io(run.outputs))]
    base_span.child_spans = [self._convert_lc_run_to_wb_span(child_run) for child_run in run.child_runs]
    base_span.span_kind = self.trace_tree.SpanKind.AGENT if 'agent' in run.name.lower() else self.trace_tree.SpanKind.CHAIN
    return base_span