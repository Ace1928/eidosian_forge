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
def _process_data(self, manager: _ExperimentManager) -> None:
    tqdm = _load_tqdm()
    results = manager.get_results()
    for item in tqdm(results):
        with self._lock:
            self._results.append(item)
    summary_scores = manager.get_summary_scores()
    with self._lock:
        self._summary_results = summary_scores