from __future__ import annotations
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
from uuid import UUID
from tenacity import RetryCallState
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
from langchain_core.tracers.schemas import Run
@staticmethod
def _add_child_run(parent_run: Run, child_run: Run) -> None:
    """Add child run to a chain run or tool run."""
    parent_run.child_runs.append(child_run)