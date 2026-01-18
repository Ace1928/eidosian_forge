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
def _set_tracing_context(context: Dict[str, Any]):
    """Set the tracing context."""
    for k, v in context.items():
        var = _CONTEXT_KEYS[k]
        var.set(v)