from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
def _collect_metrics(self) -> Tuple[Dict[str, _RowResult], Dict[str, Run]]:
    all_eval_results: dict = {}
    all_runs: dict = {}
    for c in self.configs:
        for callback in cast(list, c['callbacks']):
            if isinstance(callback, EvaluatorCallbackHandler):
                eval_results = callback.logged_eval_results
                for (_, example_id), v in eval_results.items():
                    all_eval_results.setdefault(str(example_id), {}).update({'feedback': v})
            elif isinstance(callback, LangChainTracer):
                run = callback.latest_run
                execution_time = (run.end_time - run.start_time).total_seconds() if run and run.end_time else None
                run_id = str(run.id) if run else None
                all_eval_results.setdefault(str(callback.example_id), {}).update({'execution_time': execution_time, 'run_id': run_id, 'run': run})
                all_runs[str(callback.example_id)] = run
    return (cast(Dict[str, _RowResult], all_eval_results), all_runs)