import asyncio
import time
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableConfig
class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""
    responses: List[str]
    sleep: Optional[float] = None
    i: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'fake-list'

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    async def _acall(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'responses': self.responses}