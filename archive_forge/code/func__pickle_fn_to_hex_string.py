import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def _pickle_fn_to_hex_string(fn: Callable) -> str:
    """Pickles a function and returns the hexadecimal string."""
    try:
        import cloudpickle
    except Exception as e:
        raise ValueError(f'Please install cloudpickle>=2.0.0. Error: {e}')
    try:
        return cloudpickle.dumps(fn).hex()
    except Exception as e:
        raise ValueError(f'Failed to pickle the function: {e}')