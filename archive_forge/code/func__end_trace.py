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
def _end_trace(self, run: Run) -> None:
    """End a trace for a run."""
    if not run.parent_run_id:
        self._persist_run(run)
    else:
        parent_run = self.run_map.get(str(run.parent_run_id))
        if parent_run is None:
            logger.debug(f'Parent run with UUID {run.parent_run_id} not found.')
        elif run.child_execution_order is not None and parent_run.child_execution_order is not None and (run.child_execution_order > parent_run.child_execution_order):
            parent_run.child_execution_order = run.child_execution_order
    self.run_map.pop(str(run.id))
    self._on_run_update(run)