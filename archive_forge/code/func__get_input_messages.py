from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _get_input_messages(self, input_val: Union[str, BaseMessage, Sequence[BaseMessage]]) -> List[BaseMessage]:
    from langchain_core.messages import BaseMessage
    if isinstance(input_val, str):
        from langchain_core.messages import HumanMessage
        return [HumanMessage(content=input_val)]
    elif isinstance(input_val, BaseMessage):
        return [input_val]
    elif isinstance(input_val, (list, tuple)):
        return list(input_val)
    else:
        raise ValueError(f'Expected str, BaseMessage, List[BaseMessage], or Tuple[BaseMessage]. Got {input_val}.')