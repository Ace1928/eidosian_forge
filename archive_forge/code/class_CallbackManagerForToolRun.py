from __future__ import annotations
import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
from uuid import UUID
from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState
from langchain_core.callbacks.base import (
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
class CallbackManagerForToolRun(ParentRunManager, ToolManagerMixin):
    """Callback manager for tool run."""

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running.

        Args:
            output (Any): The output of the tool.
        """
        handle_event(self.handlers, 'on_tool_end', 'ignore_agent', output, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        handle_event(self.handlers, 'on_tool_error', 'ignore_agent', error, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)