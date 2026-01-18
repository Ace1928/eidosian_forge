from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _collection_is_ready(self) -> bool:
    """Checks whether the collection for this message history is ready
        to be queried
        """
    return self.client.Collections.get(collection=self.collection).data.status == 'READY'