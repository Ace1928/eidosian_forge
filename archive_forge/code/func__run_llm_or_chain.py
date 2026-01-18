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
def _run_llm_or_chain(example: Example, config: RunnableConfig, *, llm_or_chain_factory: MCF, input_mapper: Optional[Callable[[Dict], Any]]=None) -> Union[dict, str, LLMResult, ChatResult]:
    """
    Run the Chain or language model synchronously.

    Args:
        example: The example to run.
        llm_or_chain_factory: The Chain or language model constructor to run.
        tags: Optional tags to add to the run.
        callbacks: Optional callbacks to use during the run.

    Returns:
        Union[List[dict], List[str], List[LLMResult], List[ChatResult]]:
          The outputs of the model or chain.
    """
    chain_or_llm = 'LLM' if isinstance(llm_or_chain_factory, BaseLanguageModel) else 'Chain'
    result = None
    try:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            output: Any = _run_llm(llm_or_chain_factory, example.inputs, config['callbacks'], tags=config['tags'], input_mapper=input_mapper, metadata=config.get('metadata'))
        else:
            chain = llm_or_chain_factory()
            output = _run_chain(chain, example.inputs, config['callbacks'], tags=config['tags'], input_mapper=input_mapper, metadata=config.get('metadata'))
        result = output
    except Exception as e:
        error_type = type(e).__name__
        logger.warning(f'{chain_or_llm} failed for example {example.id} with inputs {example.inputs}\nError Type: {error_type}, Message: {e}')
        result = EvalError(Error=e)
    return result