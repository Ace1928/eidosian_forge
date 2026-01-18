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
def _submit_result(self, run_id: uuid.UUID, error: Optional[str]=None) -> None:
    if error:
        self.client.create_feedback(run_id, key='pass', score=0, comment=f'Error: {repr(error)}')
    else:
        self.client.create_feedback(run_id, key='pass', score=1)