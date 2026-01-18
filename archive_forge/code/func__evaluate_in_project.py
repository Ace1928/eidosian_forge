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
def _evaluate_in_project(self, run: Run, evaluator: langsmith.RunEvaluator) -> None:
    """Evaluate the run in the project.

        Parameters
        ----------
        run : Run
            The run to be evaluated.
        evaluator : RunEvaluator
            The evaluator to use for evaluating the run.

        """
    try:
        if self.project_name is None:
            eval_result = self.client.evaluate_run(run, evaluator)
            eval_results = [eval_result]
        with tracing_v2_enabled(project_name=self.project_name, tags=['eval'], client=self.client) as cb:
            reference_example = self.client.read_example(run.reference_example_id) if run.reference_example_id else None
            evaluation_result = evaluator.evaluate_run(run, example=reference_example)
            eval_results = self._log_evaluation_feedback(evaluation_result, run, source_run_id=cb.latest_run.id if cb.latest_run else None)
    except Exception as e:
        logger.error(f'Error evaluating run {run.id} with {evaluator.__class__.__name__}: {repr(e)}', exc_info=True)
        raise e
    example_id = str(run.reference_example_id)
    with self.lock:
        for res in eval_results:
            run_id = str(getattr(res, 'target_run_id')) if hasattr(res, 'target_run_id') else str(run.id)
            self.logged_eval_results.setdefault((run_id, example_id), []).append(res)