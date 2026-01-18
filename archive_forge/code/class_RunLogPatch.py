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
class RunLogPatch:
    """Patch to the run log."""
    ops: List[Dict[str, Any]]
    'List of jsonpatch operations, which describe how to create the run state\n    from an empty dict. This is the minimal representation of the log, designed to\n    be serialized as JSON and sent over the wire to reconstruct the log on the other\n    side. Reconstruction of the state can be done with any jsonpatch-compliant library,\n    see https://jsonpatch.com for more information.'

    def __init__(self, *ops: Dict[str, Any]) -> None:
        self.ops = list(ops)

    def __add__(self, other: Union[RunLogPatch, Any]) -> RunLog:
        if type(other) == RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(None, copy.deepcopy(ops))
            return RunLog(*ops, state=state)
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __repr__(self) -> str:
        from pprint import pformat
        return f'RunLogPatch({pformat(self.ops)[1:-1]})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunLogPatch) and self.ops == other.ops