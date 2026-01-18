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
def _log_evaluation_feedback(self, evaluator_response: Union[EvaluationResult, EvaluationResults], run: Run, source_run_id: Optional[UUID]=None) -> List[EvaluationResult]:
    results = self._select_eval_results(evaluator_response)
    for res in results:
        source_info_: Dict[str, Any] = {}
        if res.evaluator_info:
            source_info_ = {**res.evaluator_info, **source_info_}
        run_id_ = getattr(res, 'target_run_id') if hasattr(res, 'target_run_id') and res.target_run_id is not None else run.id
        self.client.create_feedback(run_id_, res.key, score=res.score, value=res.value, comment=res.comment, correction=res.correction, source_info=source_info_, source_run_id=res.source_run_id or source_run_id, feedback_source_type=langsmith.schemas.FeedbackSourceType.MODEL)
    return results