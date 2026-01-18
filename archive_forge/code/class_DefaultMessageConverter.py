import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sqlalchemy import Column, Integer, Text, create_engine
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from sqlalchemy.orm import sessionmaker
class DefaultMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(session_id=session_id, message=json.dumps(message_to_dict(message)))

    def get_sql_model_class(self) -> Any:
        return self.model_class