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
def _gen_converted_chunk(message_content, message_id, finish_reason):
    transformed_response = _ChatChunkResponse(id=message_id, created=int(time.time()), model=None, choices=[_ChatChoiceDelta(index=0, delta=_ChatDeltaMessage(role='assistant', content=message_content), finish_reason=finish_reason)])
    if is_pydantic_v1:
        return json.loads(transformed_response.json())
    else:
        return transformed_response.model_dump(mode='json')