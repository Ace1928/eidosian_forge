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
def _score(self, evaluators: Sequence[RunEvaluator], max_concurrency: Optional[int]=None) -> Iterable[ExperimentResultRow]:
    """Run the evaluators on the prediction stream.

        Expects runs to be available in the manager.
        (e.g. from a previous prediction step)
        """
    if max_concurrency == 0:
        for current_results in self.get_results():
            yield self._run_evaluators(evaluators, current_results)
    else:
        with cf.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = []
            for current_results in self.get_results():
                futures.append(executor.submit(self._run_evaluators, evaluators, current_results))
            for future in cf.as_completed(futures):
                result = future.result()
                yield result