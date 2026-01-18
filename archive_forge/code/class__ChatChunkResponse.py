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
class _ChatChunkResponse(pydantic.BaseModel):
    id: Optional[str] = None
    object: Literal['chat.completion.chunk'] = 'chat.completion.chunk'
    created: int
    model: Optional[str] = None
    choices: List[_ChatChoiceDelta]