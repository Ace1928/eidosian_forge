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
def _ensure_example(func: Callable, *args: Any, langtest_extra: _UTExtra, **kwargs: Any) -> Tuple[_LangSmithTestSuite, uuid.UUID]:
    client = langtest_extra['client'] or ls_client.Client()
    output_keys = langtest_extra['output_keys']
    signature = inspect.signature(func)
    inputs: dict = rh._get_inputs_safe(signature, *args, **kwargs)
    outputs = {}
    if output_keys:
        for k in output_keys:
            outputs[k] = inputs.pop(k, None)
    test_suite = _LangSmithTestSuite.from_test(client, func)
    example_id, example_name = _get_id(func, inputs, test_suite.id)
    example_id = langtest_extra['id'] or example_id
    test_suite.sync_example(example_id, inputs, outputs, metadata={'signature': _get_test_repr(func, signature), 'name': example_name})
    return (test_suite, example_id)