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
def _run_llm(llm: BaseLanguageModel, inputs: Dict[str, Any], callbacks: Callbacks, *, tags: Optional[List[str]]=None, input_mapper: Optional[Callable[[Dict], Any]]=None, metadata: Optional[Dict[str, Any]]=None) -> Union[str, BaseMessage]:
    """
    Run the language model on the example.

    Args:
        llm: The language model to run.
        inputs: The input dictionary.
        callbacks: The callbacks to use during the run.
        tags: Optional tags to add to the run.
        input_mapper: function to map to the inputs dictionary from an Example
    Returns:
        The LLMResult or ChatResult.
    Raises:
        ValueError: If the LLM type is unsupported.
        InputFormatError: If the input format is invalid.
    """
    if input_mapper is not None:
        prompt_or_messages = input_mapper(inputs)
        if isinstance(prompt_or_messages, str) or (isinstance(prompt_or_messages, list) and all((isinstance(msg, BaseMessage) for msg in prompt_or_messages))):
            llm_output: Union[str, BaseMessage] = llm.invoke(prompt_or_messages, config=RunnableConfig(callbacks=callbacks, tags=tags or [], metadata=metadata or {}))
        else:
            raise InputFormatError(f'Input mapper returned invalid format:  {prompt_or_messages}\nExpected a single string or list of chat messages.')
    else:
        try:
            llm_prompts = _get_prompt(inputs)
            llm_output = llm.invoke(llm_prompts, config=RunnableConfig(callbacks=callbacks, tags=tags or [], metadata=metadata or {}))
        except InputFormatError:
            llm_inputs = _get_messages(inputs)
            llm_output = llm.invoke(**llm_inputs, config=RunnableConfig(callbacks=callbacks, metadata=metadata or {}))
    return llm_output