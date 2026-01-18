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
def _run_batch_evaluators(self, runs: Dict[str, Run]) -> List[dict]:
    evaluators = self.batch_evaluators
    if not evaluators:
        return []
    runs_list = [runs[str(example.id)] for example in self.examples]
    aggregate_feedback = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for evaluator in evaluators:
            try:
                result = evaluator(runs_list, self.examples)
                if isinstance(result, EvaluationResult):
                    result = result.dict()
                aggregate_feedback.append(cast(dict, result))
                executor.submit(self.client.create_feedback, **result, run_id=None, project_id=self.project.id)
            except Exception as e:
                logger.error(f'Error running batch evaluator {repr(evaluator)}: {e}')
    return aggregate_feedback