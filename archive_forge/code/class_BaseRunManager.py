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
class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(self, *, run_id: UUID, handlers: List[BaseCallbackHandler], inheritable_handlers: List[BaseCallbackHandler], parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, inheritable_tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, inheritable_metadata: Optional[Dict[str, Any]]=None) -> None:
        """Initialize the run manager.

        Args:
            run_id (UUID): The ID of the run.
            handlers (List[BaseCallbackHandler]): The list of handlers.
            inheritable_handlers (List[BaseCallbackHandler]):
                The list of inheritable handlers.
            parent_run_id (UUID, optional): The ID of the parent run.
                Defaults to None.
            tags (Optional[List[str]]): The list of tags.
            inheritable_tags (Optional[List[str]]): The list of inheritable tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            inheritable_metadata (Optional[Dict[str, Any]]): The inheritable metadata.
        """
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    @classmethod
    def get_noop_manager(cls: Type[BRM]) -> BRM:
        """Return a manager that doesn't perform any operations.

        Returns:
            BaseRunManager: The noop manager.
        """
        return cls(run_id=uuid.uuid4(), handlers=[], inheritable_handlers=[], tags=[], inheritable_tags=[], metadata={}, inheritable_metadata={})