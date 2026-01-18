from __future__ import annotations
import functools
import inspect
import os
import os.path
import sys
import threading
import traceback
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import (
from coverage.debug import short_filename, short_stack
from coverage.types import (
class SysMonitor(TracerCore):
    """Python implementation of the raw data tracer for PEP669 implementations."""

    def __init__(self, tool_id: int) -> None:
        self.data: TTraceData
        self.trace_arcs = False
        self.should_trace: Callable[[str, FrameType], TFileDisposition]
        self.should_trace_cache: dict[str, TFileDisposition | None]
        self.should_start_context: Callable[[FrameType], str | None] | None = None
        self.switch_context: Callable[[str | None], None] | None = None
        self.warn: TWarnFn
        self.myid = tool_id
        self.code_infos: dict[int, CodeInfo] = {}
        self.code_objects: list[CodeType] = []
        self.last_lines: dict[FrameType, int] = {}
        self.local_event_codes: dict[int, CodeType] = {}
        self.sysmon_on = False
        self.stats = {'starts': 0}
        self.stopped = False
        self._activity = False

    def __repr__(self) -> str:
        points = sum((len(v) for v in self.data.values()))
        files = len(self.data)
        return f'<SysMonitor at {id(self):#x}: {points} data points in {files} files>'

    @panopticon()
    def start(self) -> None:
        """Start this Tracer."""
        self.stopped = False
        assert sys_monitoring is not None
        sys_monitoring.use_tool_id(self.myid, 'coverage.py')
        register = functools.partial(sys_monitoring.register_callback, self.myid)
        events = sys_monitoring.events
        if self.trace_arcs:
            sys_monitoring.set_events(self.myid, events.PY_START | events.PY_UNWIND)
            register(events.PY_START, self.sysmon_py_start)
            register(events.PY_RESUME, self.sysmon_py_resume_arcs)
            register(events.PY_RETURN, self.sysmon_py_return_arcs)
            register(events.PY_UNWIND, self.sysmon_py_unwind_arcs)
            register(events.LINE, self.sysmon_line_arcs)
        else:
            sys_monitoring.set_events(self.myid, events.PY_START)
            register(events.PY_START, self.sysmon_py_start)
            register(events.LINE, self.sysmon_line_lines)
        sys_monitoring.restart_events()
        self.sysmon_on = True

    @panopticon()
    def stop(self) -> None:
        """Stop this Tracer."""
        if not self.sysmon_on:
            return
        assert sys_monitoring is not None
        sys_monitoring.set_events(self.myid, 0)
        self.sysmon_on = False
        for code in self.local_event_codes.values():
            sys_monitoring.set_local_events(self.myid, code, 0)
        self.local_event_codes = {}
        sys_monitoring.free_tool_id(self.myid)

    @panopticon()
    def post_fork(self) -> None:
        """The process has forked, clean up as needed."""
        self.stop()

    def activity(self) -> bool:
        """Has there been any activity?"""
        return self._activity

    def reset_activity(self) -> None:
        """Reset the activity() flag."""
        self._activity = False

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""
        return None
    if LOG:

        def callers_frame(self) -> FrameType:
            """Get the frame of the Python code we're monitoring."""
            return inspect.currentframe().f_back.f_back.f_back
    else:

        def callers_frame(self) -> FrameType:
            """Get the frame of the Python code we're monitoring."""
            return inspect.currentframe().f_back.f_back

    @panopticon('code', '@')
    def sysmon_py_start(self, code: CodeType, instruction_offset: int) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_START events."""
        self._activity = True
        self.stats['starts'] += 1
        code_info = self.code_infos.get(id(code))
        tracing_code: bool | None = None
        file_data: TTraceFileData | None = None
        if code_info is not None:
            tracing_code = code_info.tracing
            file_data = code_info.file_data
        if tracing_code is None:
            filename = code.co_filename
            disp = self.should_trace_cache.get(filename)
            if disp is None:
                frame = inspect.currentframe().f_back
                if LOG:
                    frame = frame.f_back
                disp = self.should_trace(filename, frame)
                self.should_trace_cache[filename] = disp
            tracing_code = disp.trace
            if tracing_code:
                tracename = disp.source_filename
                assert tracename is not None
                if tracename not in self.data:
                    self.data[tracename] = set()
                file_data = self.data[tracename]
                b2l = bytes_to_lines(code)
            else:
                file_data = None
                b2l = None
            self.code_infos[id(code)] = CodeInfo(tracing=tracing_code, file_data=file_data, byte_to_line=b2l)
            self.code_objects.append(code)
            if tracing_code:
                events = sys.monitoring.events
                if self.sysmon_on:
                    assert sys_monitoring is not None
                    sys_monitoring.set_local_events(self.myid, code, events.PY_RETURN | events.PY_RESUME | events.LINE)
                    self.local_event_codes[id(code)] = code
        if tracing_code and self.trace_arcs:
            frame = self.callers_frame()
            self.last_lines[frame] = -code.co_firstlineno
            return None
        else:
            return sys.monitoring.DISABLE

    @panopticon('code', '@')
    def sysmon_py_resume_arcs(self, code: CodeType, instruction_offset: int) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_RESUME events for branch coverage."""
        frame = self.callers_frame()
        self.last_lines[frame] = frame.f_lineno

    @panopticon('code', '@', None)
    def sysmon_py_return_arcs(self, code: CodeType, instruction_offset: int, retval: object) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_RETURN events for branch coverage."""
        frame = self.callers_frame()
        code_info = self.code_infos.get(id(code))
        if code_info is not None and code_info.file_data is not None:
            last_line = self.last_lines.get(frame)
            if last_line is not None:
                arc = (last_line, -code.co_firstlineno)
                cast(Set[TArc], code_info.file_data).add(arc)
        self.last_lines.pop(frame, None)

    @panopticon('code', '@', 'exc')
    def sysmon_py_unwind_arcs(self, code: CodeType, instruction_offset: int, exception: BaseException) -> MonitorReturn:
        """Handle sys.monitoring.events.PY_UNWIND events for branch coverage."""
        frame = self.callers_frame()
        last_line = self.last_lines.pop(frame, None)
        if isinstance(exception, GeneratorExit):
            return
        code_info = self.code_infos.get(id(code))
        if code_info is not None and code_info.file_data is not None:
            if last_line is not None:
                arc = (last_line, -code.co_firstlineno)
                cast(Set[TArc], code_info.file_data).add(arc)

    @panopticon('code', 'line')
    def sysmon_line_lines(self, code: CodeType, line_number: int) -> MonitorReturn:
        """Handle sys.monitoring.events.LINE events for line coverage."""
        code_info = self.code_infos[id(code)]
        if code_info.file_data is not None:
            cast(Set[TLineNo], code_info.file_data).add(line_number)
        return sys.monitoring.DISABLE

    @panopticon('code', 'line')
    def sysmon_line_arcs(self, code: CodeType, line_number: int) -> MonitorReturn:
        """Handle sys.monitoring.events.LINE events for branch coverage."""
        code_info = self.code_infos[id(code)]
        ret = None
        if code_info.file_data is not None:
            frame = self.callers_frame()
            last_line = self.last_lines.get(frame)
            if last_line is not None:
                arc = (last_line, line_number)
                cast(Set[TArc], code_info.file_data).add(arc)
            self.last_lines[frame] = line_number
        return ret