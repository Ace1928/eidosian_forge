import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
class StackSummary(list):
    """A list of FrameSummary objects, representing a stack of frames."""

    @classmethod
    def extract(klass, frame_gen, *, limit=None, lookup_lines=True, capture_locals=False):
        """Create a StackSummary from a traceback or stack object.

        :param frame_gen: A generator that yields (frame, lineno) tuples
            whose summaries are to be included in the stack.
        :param limit: None to include all frames or the number of frames to
            include.
        :param lookup_lines: If True, lookup lines for each frame immediately,
            otherwise lookup is deferred until the frame is rendered.
        :param capture_locals: If True, the local variables from each frame will
            be captured as object representations into the FrameSummary.
        """

        def extended_frame_gen():
            for f, lineno in frame_gen:
                yield (f, (lineno, None, None, None))
        return klass._extract_from_extended_frame_gen(extended_frame_gen(), limit=limit, lookup_lines=lookup_lines, capture_locals=capture_locals)

    @classmethod
    def _extract_from_extended_frame_gen(klass, frame_gen, *, limit=None, lookup_lines=True, capture_locals=False):
        if limit is None:
            limit = getattr(sys, 'tracebacklimit', None)
            if limit is not None and limit < 0:
                limit = 0
        if limit is not None:
            if limit >= 0:
                frame_gen = itertools.islice(frame_gen, limit)
            else:
                frame_gen = collections.deque(frame_gen, maxlen=-limit)
        result = klass()
        fnames = set()
        for f, (lineno, end_lineno, colno, end_colno) in frame_gen:
            co = f.f_code
            filename = co.co_filename
            name = co.co_name
            fnames.add(filename)
            linecache.lazycache(filename, f.f_globals)
            if capture_locals:
                f_locals = f.f_locals
            else:
                f_locals = None
            result.append(FrameSummary(filename, lineno, name, lookup_line=False, locals=f_locals, end_lineno=end_lineno, colno=colno, end_colno=end_colno))
        for filename in fnames:
            linecache.checkcache(filename)
        if lookup_lines:
            for f in result:
                f.line
        return result

    @classmethod
    def from_list(klass, a_list):
        """
        Create a StackSummary object from a supplied list of
        FrameSummary objects or old-style list of tuples.
        """
        result = StackSummary()
        for frame in a_list:
            if isinstance(frame, FrameSummary):
                result.append(frame)
            else:
                filename, lineno, name, line = frame
                result.append(FrameSummary(filename, lineno, name, line=line))
        return result

    def format_frame_summary(self, frame_summary):
        """Format the lines for a single FrameSummary.

        Returns a string representing one frame involved in the stack. This
        gets called for every frame to be printed in the stack summary.
        """
        row = []
        row.append('  File "{}", line {}, in {}\n'.format(frame_summary.filename, frame_summary.lineno, frame_summary.name))
        if frame_summary.line:
            stripped_line = frame_summary.line.strip()
            row.append('    {}\n'.format(stripped_line))
            line = frame_summary._original_line
            orig_line_len = len(line)
            frame_line_len = len(frame_summary.line.lstrip())
            stripped_characters = orig_line_len - frame_line_len
            if frame_summary.colno is not None and frame_summary.end_colno is not None:
                start_offset = _byte_offset_to_character_offset(line, frame_summary.colno)
                end_offset = _byte_offset_to_character_offset(line, frame_summary.end_colno)
                code_segment = line[start_offset:end_offset]
                anchors = None
                if frame_summary.lineno == frame_summary.end_lineno:
                    with suppress(Exception):
                        anchors = _extract_caret_anchors_from_line_segment(code_segment)
                else:
                    end_offset = len(line.rstrip())
                if end_offset - start_offset < len(stripped_line) or (anchors and anchors.right_start_offset - anchors.left_end_offset > 0):
                    dp_start_offset = _display_width(line, start_offset) + 1
                    dp_end_offset = _display_width(line, end_offset) + 1
                    row.append('    ')
                    row.append(' ' * (dp_start_offset - stripped_characters))
                    if anchors:
                        dp_left_end_offset = _display_width(code_segment, anchors.left_end_offset)
                        dp_right_start_offset = _display_width(code_segment, anchors.right_start_offset)
                        row.append(anchors.primary_char * dp_left_end_offset)
                        row.append(anchors.secondary_char * (dp_right_start_offset - dp_left_end_offset))
                        row.append(anchors.primary_char * (dp_end_offset - dp_start_offset - dp_right_start_offset))
                    else:
                        row.append('^' * (dp_end_offset - dp_start_offset))
                    row.append('\n')
        if frame_summary.locals:
            for name, value in sorted(frame_summary.locals.items()):
                row.append('    {name} = {value}\n'.format(name=name, value=value))
        return ''.join(row)

    def format(self):
        """Format the stack ready for printing.

        Returns a list of strings ready for printing.  Each string in the
        resulting list corresponds to a single frame from the stack.
        Each string ends in a newline; the strings may contain internal
        newlines as well, for those items with source text lines.

        For long sequences of the same frame and line, the first few
        repetitions are shown, followed by a summary line stating the exact
        number of further repetitions.
        """
        result = []
        last_file = None
        last_line = None
        last_name = None
        count = 0
        for frame_summary in self:
            formatted_frame = self.format_frame_summary(frame_summary)
            if formatted_frame is None:
                continue
            if last_file is None or last_file != frame_summary.filename or last_line is None or (last_line != frame_summary.lineno) or (last_name is None) or (last_name != frame_summary.name):
                if count > _RECURSIVE_CUTOFF:
                    count -= _RECURSIVE_CUTOFF
                    result.append(f'  [Previous line repeated {count} more time{('s' if count > 1 else '')}]\n')
                last_file = frame_summary.filename
                last_line = frame_summary.lineno
                last_name = frame_summary.name
                count = 0
            count += 1
            if count > _RECURSIVE_CUTOFF:
                continue
            result.append(formatted_frame)
        if count > _RECURSIVE_CUTOFF:
            count -= _RECURSIVE_CUTOFF
            result.append(f'  [Previous line repeated {count} more time{('s' if count > 1 else '')}]\n')
        return result