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
def _merge_test_outputs(self, batch_results: list, all_eval_results: Dict[str, _RowResult]) -> dict:
    results: dict = {}
    for example, output in zip(self.examples, batch_results):
        row_result = cast(_RowResult, all_eval_results.get(str(example.id), {}))
        results[str(example.id)] = {'input': example.inputs, 'feedback': row_result.get('feedback', []), 'execution_time': row_result.get('execution_time'), 'run_id': row_result.get('run_id')}
        if isinstance(output, EvalError):
            results[str(example.id)]['Error'] = output.Error
        else:
            results[str(example.id)]['output'] = output
        if example.outputs:
            results[str(example.id)]['reference'] = example.outputs
    return results