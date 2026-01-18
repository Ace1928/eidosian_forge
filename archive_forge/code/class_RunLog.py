from __future__ import annotations
import asyncio
import copy
import threading
from collections import defaultdict
from typing import (
from uuid import UUID
import jsonpatch  # type: ignore[import]
from typing_extensions import NotRequired, TypedDict
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.tracers.schemas import Run
class RunLog(RunLogPatch):
    """Run log."""
    state: RunState
    'Current state of the log, obtained from applying all ops in sequence.'

    def __init__(self, *ops: Dict[str, Any], state: RunState) -> None:
        super().__init__(*ops)
        self.state = state

    def __add__(self, other: Union[RunLogPatch, Any]) -> RunLog:
        if type(other) == RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(self.state, other.ops)
            return RunLog(*ops, state=state)
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __repr__(self) -> str:
        from pprint import pformat
        return f'RunLog({pformat(self.state)})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunLog):
            return False
        if self.state != other.state:
            return False
        return super().__eq__(other)