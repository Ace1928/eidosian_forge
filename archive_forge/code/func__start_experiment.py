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
def _start_experiment(client: ls_client.Client, test_suite: ls_schemas.Dataset) -> ls_schemas.TracerSession:
    experiment_name = _get_experiment_name()
    try:
        return client.create_project(experiment_name, reference_dataset_id=test_suite.id, description='Test Suite Results.', metadata={'revision_id': ls_env.get_langchain_env_var_metadata().get('revision_id')})
    except ls_utils.LangSmithConflictError:
        return client.read_project(project_name=experiment_name)