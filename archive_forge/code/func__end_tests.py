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
def _end_tests(test_suite: _LangSmithTestSuite):
    git_info = ls_env.get_git_info() or {}
    test_suite.client.update_project(test_suite.experiment_id, end_time=datetime.datetime.now(datetime.timezone.utc), metadata={**git_info, 'dataset_version': test_suite.get_version()})
    test_suite.wait()