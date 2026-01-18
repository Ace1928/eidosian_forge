from __future__ import annotations
import inspect
from typing import (
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
def _exit_history(self, run: Run, config: RunnableConfig) -> None:
    hist: BaseChatMessageHistory = config['configurable']['message_history']
    inputs = load(run.inputs)
    input_val = inputs[self.input_messages_key or 'input']
    input_messages = self._get_input_messages(input_val)
    if not self.history_messages_key:
        historic_messages = config['configurable']['message_history'].messages
        input_messages = input_messages[len(historic_messages):]
    output_val = load(run.outputs)
    output_messages = self._get_output_messages(output_val)
    hist.add_messages(input_messages + output_messages)