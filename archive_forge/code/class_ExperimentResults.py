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
class ExperimentResults:
    """Represents the results of an evaluate() call.

    This class provides an iterator interface to iterate over the experiment results
    as they become available. It also provides methods to access the experiment name,
    the number of results, and to wait for the results to be processed.

    Methods:
        experiment_name() -> str: Returns the name of the experiment.
        wait() -> None: Waits for the experiment data to be processed.
    """

    def __init__(self, experiment_manager: _ExperimentManager):
        self._manager = experiment_manager
        self._results: List[ExperimentResultRow] = []
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=lambda: self._process_data(self._manager))
        self._thread.start()

    @property
    def experiment_name(self) -> str:
        return self._manager.experiment_name

    def __iter__(self) -> Iterator[ExperimentResultRow]:
        processed_count = 0
        while True:
            with self._lock:
                if processed_count < len(self._results):
                    yield self._results[processed_count]
                    processed_count += 1
                elif not self._thread.is_alive():
                    break

    def _process_data(self, manager: _ExperimentManager) -> None:
        tqdm = _load_tqdm()
        results = manager.get_results()
        for item in tqdm(results):
            with self._lock:
                self._results.append(item)
        summary_scores = manager.get_summary_scores()
        with self._lock:
            self._summary_results = summary_scores

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return f'<ExperimentResults {self.experiment_name}>'

    def wait(self) -> None:
        """Wait for the evaluation runner to complete.

        This method blocks the current thread until the evaluation runner has
        finished its execution.
        """
        self._thread.join()