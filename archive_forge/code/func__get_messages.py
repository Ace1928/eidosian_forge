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
def _get_messages(inputs: Dict[str, Any]) -> dict:
    """Get Chat Messages from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A list of chat messages.
    Raises:
        InputFormatError: If the input format is invalid.
    """
    if not inputs:
        raise InputFormatError('Inputs should not be empty.')
    input_copy = inputs.copy()
    if 'messages' in inputs:
        input_copy['input'] = input_copy.pop('messages')
    elif len(inputs) == 1:
        input_copy['input'] = next(iter(inputs.values()))
    if 'input' in input_copy:
        raw_messages = input_copy['input']
        if isinstance(raw_messages, list) and all((isinstance(i, dict) for i in raw_messages)):
            raw_messages = [raw_messages]
        if len(raw_messages) == 1:
            input_copy['input'] = messages_from_dict(raw_messages[0])
        else:
            raise InputFormatError('Batch messages not supported. Please provide a single list of messages.')
        return input_copy
    else:
        raise InputFormatError(f"Chat Run expects single List[dict] or List[List[dict]] 'messages' input. Got {inputs}")