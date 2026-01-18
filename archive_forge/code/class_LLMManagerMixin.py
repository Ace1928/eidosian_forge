from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]]=None, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            chunk (GenerationChunk | ChatGenerationChunk): The new generated chunk,
            containing content and other information.
        """

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when LLM errors.
        Args:
            error (BaseException): The error that occurred.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """