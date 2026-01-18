from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _get_test_suite_name(func: Callable) -> str:
    test_suite_name = ls_utils.get_env_var('TEST_SUITE')
    if test_suite_name:
        return test_suite_name
    repo_name = ls_env.get_git_info()['repo_name']
    try:
        mod = inspect.getmodule(func)
        if mod:
            return f'{repo_name}.{mod.__name__}'
    except BaseException:
        logger.debug('Could not determine test suite name from file path.')
    raise ValueError('Please set the LANGSMITH_TEST_SUITE environment variable.')