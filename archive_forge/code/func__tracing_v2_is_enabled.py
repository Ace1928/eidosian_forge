from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import UUID
from langsmith import utils as ls_utils
from langsmith.run_helpers import get_run_tree_context
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import TracerSessionV1
from langchain_core.utils.env import env_var_is_set
def _tracing_v2_is_enabled() -> bool:
    return env_var_is_set('LANGCHAIN_TRACING_V2') or tracing_v2_callback_var.get() is not None or get_run_tree_context() is not None