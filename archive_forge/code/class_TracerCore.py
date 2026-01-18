from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TracerCore(Protocol):
    """Anything that can report on Python execution."""
    data: TTraceData
    trace_arcs: bool
    should_trace: Callable[[str, FrameType], TFileDisposition]
    should_trace_cache: Mapping[str, TFileDisposition | None]
    should_start_context: Callable[[FrameType], str | None] | None
    switch_context: Callable[[str | None], None] | None
    warn: TWarnFn

    def __init__(self) -> None:
        ...

    def start(self) -> TTraceFn | None:
        """Start this tracer, return a trace function if based on sys.settrace."""

    def stop(self) -> None:
        """Stop this tracer."""

    def activity(self) -> bool:
        """Has there been any activity?"""

    def reset_activity(self) -> None:
        """Reset the activity() flag."""

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""