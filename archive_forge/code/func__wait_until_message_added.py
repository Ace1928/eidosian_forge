from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _wait_until_message_added(self, message_id: str) -> None:
    """Sleeps until a message is added to the messages list"""
    self._wait_until(lambda message_id: len(self._query(f'\n                        SELECT * \n                        FROM UNNEST((\n                            SELECT {self.messages_key}\n                            FROM {self.location}\n                            WHERE _id = :session_id\n                        )) AS message\n                        WHERE message.data.additional_kwargs.id = :message_id\n                        LIMIT 1\n                    ', session_id=self.session_id, message_id=message_id)) != 0, RocksetChatMessageHistory.ADD_TIMEOUT_MS, message_id=message_id)