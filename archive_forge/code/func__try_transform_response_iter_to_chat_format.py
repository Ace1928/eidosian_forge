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
@staticmethod
def _try_transform_response_iter_to_chat_format(chunk_iter):
    from langchain_core.messages.ai import AIMessageChunk
    is_pydantic_v1 = Version(pydantic.__version__) < Version('2.0')

    def _gen_converted_chunk(message_content, message_id, finish_reason):
        transformed_response = _ChatChunkResponse(id=message_id, created=int(time.time()), model=None, choices=[_ChatChoiceDelta(index=0, delta=_ChatDeltaMessage(role='assistant', content=message_content), finish_reason=finish_reason)])
        if is_pydantic_v1:
            return json.loads(transformed_response.json())
        else:
            return transformed_response.model_dump(mode='json')

    def _convert(chunk):
        if isinstance(chunk, str):
            message_content = chunk
            message_id = None
            finish_reason = None
        elif isinstance(chunk, AIMessageChunk):
            message_content = chunk.content
            message_id = getattr(chunk, 'id', None)
            if (response_metadata := getattr(chunk, 'response_metadata', None)):
                finish_reason = response_metadata.get('finish_reason')
            else:
                finish_reason = None
        elif isinstance(chunk, AIMessage):
            message_content = chunk.content
            message_id = getattr(chunk, 'id', None)
            finish_reason = 'stop'
        else:
            return chunk
        return _gen_converted_chunk(message_content, message_id=message_id, finish_reason=finish_reason)
    return map(_convert, chunk_iter)