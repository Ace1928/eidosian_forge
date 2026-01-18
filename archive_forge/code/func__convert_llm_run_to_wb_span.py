from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def _convert_llm_run_to_wb_span(self, run: Run) -> 'Span':
    """Converts a LangChain LLM Run into a W&B Trace Span.
        :param run: The LangChain LLM Run to convert.
        :return: The converted W&B Trace Span.
        """
    base_span = self._convert_run_to_wb_span(run)
    if base_span.attributes is None:
        base_span.attributes = {}
    base_span.attributes['llm_output'] = (run.outputs or {}).get('llm_output', {})
    base_span.results = [self.trace_tree.Result(inputs={'prompt': prompt}, outputs={f'gen_{g_i}': gen['text'] for g_i, gen in enumerate(run.outputs['generations'][ndx])} if run.outputs is not None and len(run.outputs['generations']) > ndx and (len(run.outputs['generations'][ndx]) > 0) else None) for ndx, prompt in enumerate(run.inputs['prompts'] or [])]
    base_span.span_kind = self.trace_tree.SpanKind.LLM
    return base_span