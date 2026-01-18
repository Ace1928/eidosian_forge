from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _convert_run_to_wb_span(self, run: Run) -> 'Span':
    """Base utility to create a span from a run.
        :param run: The run to convert.
        :return: The converted Span.
        """
    attributes = {**run.extra} if run.extra else {}
    attributes['execution_order'] = run.execution_order
    return self.trace_tree.Span(span_id=str(run.id) if run.id is not None else None, name=run.name, start_time_ms=int(run.start_time.timestamp() * 1000), end_time_ms=int(run.end_time.timestamp() * 1000) if run.end_time is not None else None, status_code=self.trace_tree.StatusCode.SUCCESS if run.error is None else self.trace_tree.StatusCode.ERROR, status_message=run.error, attributes=attributes)