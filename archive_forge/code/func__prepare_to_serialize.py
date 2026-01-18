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
def _prepare_to_serialize(self, response: dict):
    """
        Converts LangChain objects to JSON-serializable formats.
        """
    from langchain.load.dump import dumps
    if 'intermediate_steps' in response:
        steps = response['intermediate_steps']
        if isinstance(steps, tuple) and len(steps) == 2 and isinstance(steps[0], AgentAction) and isinstance(steps[1], str):
            response['intermediate_steps'] = [{'tool': agent.tool, 'tool_input': agent.tool_input, 'log': agent.log, 'result': result} for agent, result in response['intermediate_steps']]
        else:
            try:
                response['intermediate_steps'] = dumps(steps)
            except Exception as e:
                _logger.warning(f'Failed to serialize intermediate steps: {e!r}')
    if 'source_documents' in response:
        response['source_documents'] = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in response['source_documents']]