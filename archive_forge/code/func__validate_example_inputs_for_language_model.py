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
def _validate_example_inputs_for_language_model(first_example: Example, input_mapper: Optional[Callable[[Dict], Any]]) -> None:
    if input_mapper:
        prompt_input = input_mapper(first_example.inputs)
        if not isinstance(prompt_input, str) and (not (isinstance(prompt_input, list) and all((isinstance(msg, BaseMessage) for msg in prompt_input)))):
            raise InputFormatError(f'When using an input_mapper to prepare dataset example inputs for an LLM or chat model, the output must a single string or a list of chat messages.\nGot: {prompt_input} of type {type(prompt_input)}.')
    else:
        try:
            _get_prompt(first_example.inputs)
        except InputFormatError:
            try:
                _get_messages(first_example.inputs)
            except InputFormatError:
                raise InputFormatError(f'Example inputs do not match language model input format. Expected a dictionary with messages or a single prompt. Got: {first_example.inputs} Please update your dataset OR provide an input_mapper to convert the example.inputs to a compatible format for the llm or chat model you wish to evaluate.')