from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _wait_until_collection_created(self) -> None:
    """Sleeps until the collection for this message history is ready
        to be queried
        """
    self._wait_until(lambda: self._collection_is_ready(), RocksetChatMessageHistory.CREATE_TIMEOUT_MS)