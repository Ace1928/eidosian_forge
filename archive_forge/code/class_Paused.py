from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('Debugger.paused')
@dataclass
class Paused:
    """
    Fired when the virtual machine stopped on breakpoint or exception or any other stop criteria.
    """
    call_frames: typing.List[CallFrame]
    reason: str
    data: typing.Optional[dict]
    hit_breakpoints: typing.Optional[typing.List[str]]
    async_stack_trace: typing.Optional[runtime.StackTrace]
    async_stack_trace_id: typing.Optional[runtime.StackTraceId]
    async_call_stack_trace_id: typing.Optional[runtime.StackTraceId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Paused:
        return cls(call_frames=[CallFrame.from_json(i) for i in json['callFrames']], reason=str(json['reason']), data=dict(json['data']) if 'data' in json else None, hit_breakpoints=[str(i) for i in json['hitBreakpoints']] if 'hitBreakpoints' in json else None, async_stack_trace=runtime.StackTrace.from_json(json['asyncStackTrace']) if 'asyncStackTrace' in json else None, async_stack_trace_id=runtime.StackTraceId.from_json(json['asyncStackTraceId']) if 'asyncStackTraceId' in json else None, async_call_stack_trace_id=runtime.StackTraceId.from_json(json['asyncCallStackTraceId']) if 'asyncCallStackTraceId' in json else None)