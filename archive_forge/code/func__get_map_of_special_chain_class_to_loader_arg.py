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
def _get_map_of_special_chain_class_to_loader_arg():
    import langchain
    from mlflow.langchain.retriever_chain import _RetrieverChain
    class_name_to_loader_arg = {'langchain.chains.RetrievalQA': 'retriever', 'langchain.chains.APIChain': 'requests_wrapper', 'langchain.chains.HypotheticalDocumentEmbedder': 'embeddings'}
    if version.parse(langchain.__version__) <= version.parse('0.0.246'):
        class_name_to_loader_arg['langchain.chains.SQLDatabaseChain'] = 'database'
    elif find_spec('langchain_experimental'):
        class_name_to_loader_arg['langchain_experimental.sql.SQLDatabaseChain'] = 'database'
    class_to_loader_arg = {_RetrieverChain: 'retriever'}
    for class_name, loader_arg in class_name_to_loader_arg.items():
        try:
            cls = _get_class_from_string(class_name)
            class_to_loader_arg[cls] = loader_arg
        except Exception:
            logger.warning("Unexpected import failure for class '%s'. Please file an issue at https://github.com/mlflow/mlflow/issues/.", class_name, exc_info=True)
    return class_to_loader_arg