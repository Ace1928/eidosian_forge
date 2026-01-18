from abc import abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.llms.gradient_ai import TrainResult
class Memorize(BaseTool):
    """Tool that trains a language model."""
    name: str = 'memorize'
    description: str = "Useful whenever you observed novel information from previous conversation history, i.e., another tool's action outputs or human comments. The action input should include observed information in detail, then the tool will fine-tune yourself to remember it."
    llm: TrainableLLM = Field()

    def _run(self, information_to_learn: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        train_result = self.llm.train_unsupervised((information_to_learn,))
        return f'Train complete. Loss: {train_result['loss']}'

    async def _arun(self, information_to_learn: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        train_result = await self.llm.atrain_unsupervised((information_to_learn,))
        return f'Train complete. Loss: {train_result['loss']}'