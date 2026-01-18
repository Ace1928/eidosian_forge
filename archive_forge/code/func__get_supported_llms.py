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
@lru_cache
def _get_supported_llms():
    import langchain.chat_models
    import langchain.llms
    llms = {langchain.llms.openai.OpenAI, langchain.llms.huggingface_hub.HuggingFaceHub}
    if hasattr(langchain.llms, 'Databricks'):
        llms.add(langchain.llms.Databricks)
    if hasattr(langchain.llms, 'Mlflow'):
        llms.add(langchain.llms.Mlflow)
    if hasattr(langchain.chat_models, 'ChatDatabricks'):
        llms.add(langchain.chat_models.ChatDatabricks)
    if hasattr(langchain.chat_models, 'ChatMlflow'):
        llms.add(langchain.chat_models.ChatMlflow)
    if hasattr(langchain.chat_models, 'ChatOpenAI'):
        llms.add(langchain.chat_models.ChatOpenAI)
    if hasattr(langchain.chat_models, 'AzureChatOpenAI'):
        llms.add(langchain.chat_models.AzureChatOpenAI)
    return llms