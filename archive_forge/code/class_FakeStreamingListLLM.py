import asyncio
import time
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableConfig
class FakeStreamingListLLM(FakeListLLM):
    """Fake streaming list LLM for testing purposes."""

    def stream(self, input: LanguageModelInput, config: Optional[RunnableConfig]=None, *, stop: Optional[List[str]]=None, **kwargs: Any) -> Iterator[str]:
        result = self.invoke(input, config)
        for c in result:
            if self.sleep is not None:
                time.sleep(self.sleep)
            yield c

    async def astream(self, input: LanguageModelInput, config: Optional[RunnableConfig]=None, *, stop: Optional[List[str]]=None, **kwargs: Any) -> AsyncIterator[str]:
        result = await self.ainvoke(input, config)
        for c in result:
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            yield c