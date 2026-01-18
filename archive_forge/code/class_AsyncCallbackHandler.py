from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler that handles callbacks from LangChain."""

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.
        """

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `on_chat_model_start`')

    async def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]]=None, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        """Run when chain starts running."""

    async def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when chain ends running."""

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when chain errors."""

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, inputs: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        """Run when tool starts running."""

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when tool ends running."""

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run when tool errors."""

    async def on_text(self, text: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    async def on_retry(self, retry_state: RetryCallState, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run on a retry event."""

    async def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on agent action."""

    async def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on agent end."""

    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        """Run on retriever start."""

    async def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on retriever end."""

    async def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Run on retriever error."""