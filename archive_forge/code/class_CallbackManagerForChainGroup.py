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
class CallbackManagerForChainGroup(CallbackManager):
    """Callback manager for the chain group."""

    def __init__(self, handlers: List[BaseCallbackHandler], inheritable_handlers: Optional[List[BaseCallbackHandler]]=None, parent_run_id: Optional[UUID]=None, *, parent_run_manager: CallbackManagerForChainRun, **kwargs: Any) -> None:
        super().__init__(handlers, inheritable_handlers, parent_run_id, **kwargs)
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> CallbackManagerForChainGroup:
        return self.__class__(handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata, parent_run_manager=self.parent_run_manager)

    def on_chain_end(self, outputs: Union[Dict[str, Any], Any], **kwargs: Any) -> None:
        """Run when traced chain group ends.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        self.ended = True
        return self.parent_run_manager.on_chain_end(outputs, **kwargs)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        self.ended = True
        return self.parent_run_manager.on_chain_error(error, **kwargs)