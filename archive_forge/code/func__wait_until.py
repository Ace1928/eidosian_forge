from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _wait_until(self, method: Callable, timeout: int, **method_params: Any) -> None:
    """Sleeps until meth() evaluates to true. Passes kwargs into
        meth.
        """
    start = datetime.now()
    while not method(**method_params):
        curr = datetime.now()
        if (curr - start).total_seconds() * 1000 > timeout:
            raise TimeoutError(f'{method} timed out at {timeout} ms')
        sleep(RocksetChatMessageHistory.SLEEP_INTERVAL_MS / 1000)