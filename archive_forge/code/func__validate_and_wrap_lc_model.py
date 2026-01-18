import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import types
import warnings
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple
import cloudpickle
import yaml
from packaging import version
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.class_utils import _get_class_from_string
def _validate_and_wrap_lc_model(lc_model, loader_fn):
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.chains.llm
    import langchain.llms.huggingface_hub
    import langchain.llms.openai
    import langchain.schema
    if isinstance(lc_model, str):
        if os.path.basename(os.path.abspath(lc_model)) != 'chain.py':
            raise mlflow.MlflowException.invalid_parameter_value(f'If {lc_model} is a string, it must be the path to a file named `chain.py` on the local filesystem.')
        return lc_model
    if not isinstance(lc_model, supported_lc_types()):
        raise mlflow.MlflowException.invalid_parameter_value(_UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(lc_model).__name__))
    _SUPPORTED_LLMS = _get_supported_llms()
    if isinstance(lc_model, langchain.chains.llm.LLMChain) and (not any((isinstance(lc_model.llm, supported_llm) for supported_llm in _SUPPORTED_LLMS))):
        logger.warning(_UNSUPPORTED_LLM_WARNING_MESSAGE, type(lc_model.llm).__name__)
    if isinstance(lc_model, langchain.agents.agent.AgentExecutor) and (not any((isinstance(lc_model.agent.llm_chain.llm, supported_llm) for supported_llm in _SUPPORTED_LLMS))):
        logger.warning(_UNSUPPORTED_LLM_WARNING_MESSAGE, type(lc_model.agent.llm_chain.llm).__name__)
    if (special_chain_info := _get_special_chain_info_or_none(lc_model)):
        if loader_fn is None:
            raise mlflow.MlflowException.invalid_parameter_value(f'For {type(lc_model).__name__} models, a `loader_fn` must be provided.')
        if not isinstance(loader_fn, types.FunctionType):
            raise mlflow.MlflowException.invalid_parameter_value('The `loader_fn` must be a function that returns a {loader_arg}.'.format(loader_arg=special_chain_info.loader_arg))
    if isinstance(lc_model, langchain.schema.BaseRetriever):
        from mlflow.langchain.retriever_chain import _RetrieverChain
        if loader_fn is None:
            raise mlflow.MlflowException.invalid_parameter_value(f'For {type(lc_model).__name__} models, a `loader_fn` must be provided.')
        if not isinstance(loader_fn, types.FunctionType):
            raise mlflow.MlflowException.invalid_parameter_value('The `loader_fn` must be a function that returns a retriever.')
        lc_model = _RetrieverChain(retriever=lc_model)
    return lc_model