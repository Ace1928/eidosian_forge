import os
from typing import Any, Dict, List
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
def _log_conversation(self) -> None:
    """Log the conversation to the context API."""
    if len(self.messages) == 0:
        return
    self.client.log.conversation_upsert(body={'conversation': self.conversation_model(messages=self.messages, metadata=self.metadata)})
    self.messages = []
    self.metadata = {}