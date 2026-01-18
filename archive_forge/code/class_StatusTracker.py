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
@dataclass
class StatusTracker:
    """
    Stores metadata about the script's progress. Only one instance is created.
    """
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_api_errors: int = 0
    lock: threading.Lock = threading.Lock()

    def start_task(self):
        with self.lock:
            self.num_tasks_started += 1
            self.num_tasks_in_progress += 1

    def complete_task(self, *, success: bool):
        with self.lock:
            self.num_tasks_in_progress -= 1
            if success:
                self.num_tasks_succeeded += 1
            else:
                self.num_tasks_failed += 1

    def increment_num_api_errors(self):
        with self.lock:
            self.num_api_errors += 1