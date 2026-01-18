from __future__ import annotations
import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID
import langsmith
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langchain_core.tracers import langchain as langchain_tracer
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import _get_executor
from langchain_core.tracers.schemas import Run
def _select_eval_results(self, results: Union[EvaluationResult, EvaluationResults]) -> List[EvaluationResult]:
    if isinstance(results, EvaluationResult):
        results_ = [results]
    elif isinstance(results, dict) and 'results' in results:
        results_ = cast(List[EvaluationResult], results['results'])
    else:
        raise TypeError(f'Invalid evaluation result type {type(results)}. Expected EvaluationResult or EvaluationResults.')
    return results_