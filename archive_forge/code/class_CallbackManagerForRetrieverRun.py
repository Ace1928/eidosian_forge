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
class CallbackManagerForRetrieverRun(ParentRunManager, RetrieverManagerMixin):
    """Callback manager for retriever run."""

    def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> None:
        """Run when retriever ends running."""
        handle_event(self.handlers, 'on_retriever_end', 'ignore_retriever', documents, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when retriever errors."""
        handle_event(self.handlers, 'on_retriever_error', 'ignore_retriever', error, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)