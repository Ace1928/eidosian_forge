from __future__ import annotations
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def add_ai_message(self, message: str, metadata: Optional[Dict[str, Any]]=None) -> None:
    """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
            metadata: Optional metadata to attach to the message.
        """
    self.add_message(AIMessage(content=message), metadata=metadata)