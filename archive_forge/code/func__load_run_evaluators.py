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
def _load_run_evaluators(config: smith_eval.RunEvalConfig, run_type: str, data_type: DataType, example_outputs: Optional[List[str]], run_inputs: Optional[List[str]], run_outputs: Optional[List[str]]) -> List[RunEvaluator]:
    """
    Load run evaluators from a configuration.

    Args:
        config: Configuration for the run evaluators.

    Returns:
        A list of run evaluators.
    """
    run_evaluators = []
    input_key, prediction_key, reference_key = (None, None, None)
    if config.evaluators or (config.custom_evaluators and any([isinstance(e, StringEvaluator) for e in config.custom_evaluators])):
        input_key, prediction_key, reference_key = _get_keys(config, run_inputs, run_outputs, example_outputs)
    for eval_config in config.evaluators:
        run_evaluator = _construct_run_evaluator(eval_config, config.eval_llm, run_type, data_type, example_outputs, reference_key, input_key, prediction_key)
        run_evaluators.append(run_evaluator)
    custom_evaluators = config.custom_evaluators or []
    for custom_evaluator in custom_evaluators:
        if isinstance(custom_evaluator, RunEvaluator):
            run_evaluators.append(custom_evaluator)
        elif isinstance(custom_evaluator, StringEvaluator):
            run_evaluators.append(smith_eval.StringRunEvaluatorChain.from_run_and_data_type(custom_evaluator, run_type, data_type, input_key=input_key, prediction_key=prediction_key, reference_key=reference_key))
        elif callable(custom_evaluator):
            run_evaluators.append(run_evaluator_dec(custom_evaluator))
        else:
            raise ValueError(f'Unsupported custom evaluator: {custom_evaluator}. Expected RunEvaluator or StringEvaluator.')
    return run_evaluators