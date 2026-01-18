from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _get_parameter_names(callable_: GetSessionHistoryCallable) -> List[str]:
    """Get the parameter names of the callable."""
    sig = inspect.signature(callable_)
    return list(sig.parameters.keys())