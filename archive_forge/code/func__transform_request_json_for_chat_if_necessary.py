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
def _transform_request_json_for_chat_if_necessary(request_json, lc_model):
    """
        Returns:
            A 2-element tuple containing:

                1. The new request.
                2. A boolean indicating whether or not the request was transformed from the OpenAI
                chat format.
        """
    input_fields = APIRequest._get_lc_model_input_fields(lc_model)
    if 'messages' in input_fields:
        return (request_json, False)

    def json_dict_might_be_chat_request(json_message: Dict):
        return isinstance(json_message, dict) and 'messages' in json_message and (len(json_message) == 1)
    if isinstance(request_json, dict) and json_dict_might_be_chat_request(request_json):
        try:
            return (APIRequest._convert_chat_request_or_throw(request_json), True)
        except pydantic.ValidationError:
            return (request_json, False)
    elif isinstance(request_json, list) and all((json_dict_might_be_chat_request(json) for json in request_json)):
        try:
            return ([APIRequest._convert_chat_request_or_throw(json_dict) for json_dict in request_json], True)
        except pydantic.ValidationError:
            return (request_json, False)
    else:
        return (request_json, False)