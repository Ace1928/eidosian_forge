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
def _patch_loader(loader_func: Callable) -> Callable:
    """
    Patch LangChain loader function like load_chain() to handle the breaking change introduced in
    LangChain 0.1.12.

    Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization
    requires the `allow_dangerous_deserialization` flag to be set to True, for security reasons.
    However, this flag could not be specified via the LangChain's loading API like load_chain(),
    load_llm(), until LangChain 0.1.14. As a result, such module cannot be loaded with MLflow
    with earlier version of LangChain and we have to tell the user to upgrade LangChain to 0.0.14
    or above.

    Args:
        loader_func: The LangChain loader function to be patched e.g. load_chain().

    Returns:
        The patched loader function.
    """
    if not IS_PICKLE_SERIALIZATION_RESTRICTED:
        return loader_func
    import langchain
    if Version(langchain.__version__) >= Version('0.1.14'):

        def patched_loader(*args, **kwargs):
            return loader_func(*args, **kwargs, allow_dangerous_deserialization=True)
    else:

        def patched_loader(*args, **kwargs):
            try:
                return loader_func(*args, **kwargs)
            except ValueError as e:
                if 'This code relies on the pickle module' in str(e):
                    raise MlflowException('Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization requires the `allow_dangerous_deserialization` flag to be set to True when loading. However, this flag is not supported by the installed version of LangChain. Please upgrade LangChain to 0.1.14 or above by running `pip install langchain>=0.1.14`.', error_code=INTERNAL_ERROR) from e
                else:
                    raise
    return patched_loader