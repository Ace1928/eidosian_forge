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
def _get_lc_model_input_fields(lc_model) -> Set:
    try:
        if hasattr(lc_model, 'input_schema') and callable(lc_model.input_schema):
            return set(lc_model.input_schema().__fields__)
    except Exception as e:
        _logger.debug(f'Unexpected exception while checking LangChain input schema for request transformation: {e}')
    return set()