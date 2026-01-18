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
class CallbackManager(BaseCallbackManager):
    """Callback manager that handles callbacks from LangChain."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], run_id: Optional[UUID]=None, **kwargs: Any) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The list of prompts.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                prompt as an LLM run.
        """
        managers = []
        for i, prompt in enumerate(prompts):
            run_id_ = run_id if i == 0 and run_id is not None else uuid.uuid4()
            handle_event(self.handlers, 'on_llm_start', 'ignore_llm', serialized, [prompt], run_id=run_id_, parent_run_id=self.parent_run_id, tags=self.tags, metadata=self.metadata, **kwargs)
            managers.append(CallbackManagerForLLMRun(run_id=run_id_, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata))
        return managers

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], run_id: Optional[UUID]=None, **kwargs: Any) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            messages (List[List[BaseMessage]]): The list of messages.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                list of messages as an LLM run.
        """
        managers = []
        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid.uuid4()
            handle_event(self.handlers, 'on_chat_model_start', 'ignore_chat_model', serialized, [message_list], run_id=run_id_, parent_run_id=self.parent_run_id, tags=self.tags, metadata=self.metadata, **kwargs)
            managers.append(CallbackManagerForLLMRun(run_id=run_id_, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata))
        return managers

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Union[Dict[str, Any], Any], run_id: Optional[UUID]=None, **kwargs: Any) -> CallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Union[Dict[str, Any], Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            CallbackManagerForChainRun: The callback manager for the chain run.
        """
        if run_id is None:
            run_id = uuid.uuid4()
        handle_event(self.handlers, 'on_chain_start', 'ignore_chain', serialized, inputs, run_id=run_id, parent_run_id=self.parent_run_id, tags=self.tags, metadata=self.metadata, **kwargs)
        return CallbackManagerForChainRun(run_id=run_id, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, run_id: Optional[UUID]=None, parent_run_id: Optional[UUID]=None, inputs: Optional[Dict[str, Any]]=None, **kwargs: Any) -> CallbackManagerForToolRun:
        """Run when tool starts running.

        Args:
            serialized: Serialized representation of the tool.
            input_str: The  input to the tool as a string.
                Non-string inputs are cast to strings.
            run_id: ID for the run. Defaults to None.
            parent_run_id: The ID of the parent run. Defaults to None.
            inputs: The original input to the tool if provided.
                Recommended for usage instead of input_str when the original
                input is needed.
                If provided, the inputs are expected to be formatted as a dict.
                The keys will correspond to the named-arguments in the tool.

        Returns:
            CallbackManagerForToolRun: The callback manager for the tool run.
        """
        if run_id is None:
            run_id = uuid.uuid4()
        handle_event(self.handlers, 'on_tool_start', 'ignore_agent', serialized, input_str, run_id=run_id, parent_run_id=self.parent_run_id, tags=self.tags, metadata=self.metadata, inputs=inputs, **kwargs)
        return CallbackManagerForToolRun(run_id=run_id, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata)

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, run_id: Optional[UUID]=None, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> CallbackManagerForRetrieverRun:
        """Run when retriever starts running."""
        if run_id is None:
            run_id = uuid.uuid4()
        handle_event(self.handlers, 'on_retriever_start', 'ignore_retriever', serialized, query, run_id=run_id, parent_run_id=self.parent_run_id, tags=self.tags, metadata=self.metadata, **kwargs)
        return CallbackManagerForRetrieverRun(run_id=run_id, handlers=self.handlers, inheritable_handlers=self.inheritable_handlers, parent_run_id=self.parent_run_id, tags=self.tags, inheritable_tags=self.inheritable_tags, metadata=self.metadata, inheritable_metadata=self.inheritable_metadata)

    @classmethod
    def configure(cls, inheritable_callbacks: Callbacks=None, local_callbacks: Callbacks=None, verbose: bool=False, inheritable_tags: Optional[List[str]]=None, local_tags: Optional[List[str]]=None, inheritable_metadata: Optional[Dict[str, Any]]=None, local_metadata: Optional[Dict[str, Any]]=None) -> CallbackManager:
        """Configure the callback manager.

        Args:
            inheritable_callbacks (Optional[Callbacks], optional): The inheritable
                callbacks. Defaults to None.
            local_callbacks (Optional[Callbacks], optional): The local callbacks.
                Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            inheritable_tags (Optional[List[str]], optional): The inheritable tags.
                Defaults to None.
            local_tags (Optional[List[str]], optional): The local tags.
                Defaults to None.
            inheritable_metadata (Optional[Dict[str, Any]], optional): The inheritable
                metadata. Defaults to None.
            local_metadata (Optional[Dict[str, Any]], optional): The local metadata.
                Defaults to None.

        Returns:
            CallbackManager: The configured callback manager.
        """
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose, inheritable_tags, local_tags, inheritable_metadata, local_metadata)