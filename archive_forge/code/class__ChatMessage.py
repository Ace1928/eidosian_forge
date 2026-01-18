from __future__ import annotations
import json
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union
import langchain.chains
import pydantic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
class _ChatMessage(pydantic.BaseModel, extra='forbid'):
    role: str
    content: str

    def to_langchain_message(self) -> LangChainChatMessage:
        if self.role == 'system':
            return SystemMessage(content=self.content)
        elif self.role == 'assistant':
            return AIMessage(content=self.content)
        elif self.role == 'user':
            return HumanMessage(content=self.content)
        else:
            raise MlflowException.invalid_parameter_value(f'Unrecognized chat message role: {self.role}')