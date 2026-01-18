from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _get_output_messages(self, output_val: Union[str, BaseMessage, Sequence[BaseMessage], dict]) -> List[BaseMessage]:
    from langchain_core.messages import BaseMessage
    if isinstance(output_val, dict):
        output_val = output_val[self.output_messages_key or 'output']
    if isinstance(output_val, str):
        from langchain_core.messages import AIMessage
        return [AIMessage(content=output_val)]
    elif isinstance(output_val, BaseMessage):
        return [output_val]
    elif isinstance(output_val, (list, tuple)):
        return list(output_val)
    else:
        raise ValueError()