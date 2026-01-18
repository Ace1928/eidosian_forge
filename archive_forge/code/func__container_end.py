from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
def _container_end(container: _TraceableContainer, outputs: Optional[Any]=None, error: Optional[str]=None):
    """End the run."""
    run_tree = container.get('new_run')
    if run_tree is None:
        return
    outputs_ = outputs if isinstance(outputs, dict) else {'output': outputs}
    run_tree.end(outputs=outputs_, error=error)
    run_tree.patch()
    if error:
        try:
            LOGGER.info(f'See trace: {run_tree.get_url()}')
        except Exception:
            pass
    on_end = container.get('on_end')
    if on_end is not None and callable(on_end):
        try:
            on_end(run_tree)
        except Exception as e:
            LOGGER.warning(f'Failed to run on_end function: {e}')