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
def _get_trace_callbacks(project_name: Optional[str]=None, example_id: Optional[Union[str, UUID]]=None, callback_manager: Optional[Union[CallbackManager, AsyncCallbackManager]]=None) -> Callbacks:
    if _tracing_v2_is_enabled():
        project_name_ = project_name or _get_tracer_project()
        tracer = tracing_v2_callback_var.get() or LangChainTracer(project_name=project_name_, example_id=example_id)
        if callback_manager is None:
            from langchain_core.callbacks.base import Callbacks
            cb = cast(Callbacks, [tracer])
        else:
            if not any((isinstance(handler, LangChainTracer) for handler in callback_manager.handlers)):
                callback_manager.add_handler(tracer, True)
            cb = callback_manager
    else:
        cb = None
    return cb