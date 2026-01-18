import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sqlalchemy import Column, Integer, Text, create_engine
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from sqlalchemy.orm import sessionmaker
class BaseMessageConverter(ABC):
    """Convert BaseMessage to the SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """Convert a SQLAlchemy model to a BaseMessage instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        """Convert a BaseMessage instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError