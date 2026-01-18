from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
def _apply_summary_evaluators(self, summary_evaluators: Sequence[SUMMARY_EVALUATOR_T]) -> Generator[EvaluationResults, None, None]:
    runs, examples = ([], [])
    for run, example in zip(self.runs, self.examples):
        runs.append(run)
        examples.append(example)
    aggregate_feedback = []
    with cf.ThreadPoolExecutor() as executor:
        project_id = self._get_experiment().id
        current_context = rh.get_tracing_context()
        metadata = {**(current_context['metadata'] or {}), **{'experiment': self.experiment_name, 'experiment_id': project_id}}
        with rh.tracing_context(**{**current_context, 'project_name': 'evaluators', 'metadata': metadata}):
            for evaluator in summary_evaluators:
                try:
                    summary_eval_result = evaluator(runs, examples)
                    flattened_results = self.client._select_eval_results(summary_eval_result, fn_name=evaluator.__name__)
                    aggregate_feedback.extend(flattened_results)
                    for result in flattened_results:
                        feedback = result.dict(exclude={'target_run_id'})
                        evaluator_info = feedback.pop('evaluator_info', None)
                        executor.submit(self.client.create_feedback, **feedback, run_id=None, project_id=project_id, source_info=evaluator_info)
                except Exception as e:
                    logger.error(f'Error running summary evaluator {repr(evaluator)}: {e}')
    yield {'results': aggregate_feedback}