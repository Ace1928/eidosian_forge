from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY
def _get_messages_from_run_dict(messages: List[dict]) -> List[BaseMessage]:
    if not messages:
        return []
    first_message = messages[0]
    if 'lc' in first_message:
        return [load(dumpd(message)) for message in messages]
    else:
        return messages_from_dict(messages)