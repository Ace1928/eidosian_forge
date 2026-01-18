import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
def _extract_frames(self, tb, is_first, *, limit=None, from_decorator=False):
    frames, final_source = ([], None)
    if tb is None or (limit is not None and limit <= 0):
        return (frames, final_source)

    def is_valid(frame):
        return frame.f_code.co_filename != self._hidden_frames_filename

    def get_info(frame, lineno):
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        source = linecache.getline(filename, lineno).strip()
        return (filename, lineno, function, source)
    infos = []
    if is_valid(tb.tb_frame):
        infos.append((get_info(tb.tb_frame, tb.tb_lineno), tb.tb_frame))
    get_parent_only = from_decorator and (not self._backtrace)
    if self._backtrace and is_first or get_parent_only:
        frame = tb.tb_frame.f_back
        while frame:
            if is_valid(frame):
                infos.insert(0, (get_info(frame, frame.f_lineno), frame))
                if get_parent_only:
                    break
            frame = frame.f_back
        if infos and (not get_parent_only):
            (filename, lineno, function, source), frame = infos[-1]
            function += self._catch_point_identifier
            infos[-1] = ((filename, lineno, function, source), frame)
    tb = tb.tb_next
    while tb:
        if is_valid(tb.tb_frame):
            infos.append((get_info(tb.tb_frame, tb.tb_lineno), tb.tb_frame))
        tb = tb.tb_next
    if limit is not None:
        infos = infos[-limit:]
    for (filename, lineno, function, source), frame in infos:
        final_source = source
        if source:
            colorize = self._colorize and self._is_file_mine(filename)
            lines = []
            if colorize:
                lines.append(self._syntax_highlighter.highlight(source))
            else:
                lines.append(source)
            if self._diagnose:
                relevant_values = self._get_relevant_values(source, frame)
                values = self._format_relevant_values(list(relevant_values), colorize)
                lines += list(values)
            source = '\n    '.join(lines)
        frames.append((filename, lineno, function, source))
    return (frames, final_source)