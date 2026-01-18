from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
class CursorAwareWindow(BaseWindow, ContextManager['CursorAwareWindow']):
    """
    Renders to the normal terminal screen and
    can find the location of the cursor.

    Note:
        The context of the CursorAwareWindow
        object must be entered before calling any of its methods.

        Within the context of CursorAwareWindow, refrain from writing to
        its out_stream; cached writes will be inaccurate and calculating
        cursor depends on cursor not having moved since the last render.
        Only use the render_to_terminal interface for moving the cursor.
    """

    def __init__(self, out_stream: Optional[IO]=None, in_stream: Optional[IO]=None, keep_last_line: bool=False, hide_cursor: bool=True, extra_bytes_callback: Optional[Callable[[bytes], None]]=None):
        """Constructs a CursorAwareWindow

        Args:
            out_stream (file): Defaults to sys.__stdout__
            in_stream (file): Defaults to sys.__stdin__
            keep_last_line (bool): Causes the cursor to be moved down one line
                on leaving context
            hide_cursor (bool): Hides cursor while in context
            extra_bytes_callback (f(bytes) -> None): Will be called with extra
                bytes inadvertently read in get_cursor_position(). If not
                provided, a ValueError will be raised when this occurs.
        """
        super().__init__(out_stream=out_stream, hide_cursor=hide_cursor)
        if in_stream is None:
            in_stream = sys.__stdin__
        self.in_stream = in_stream
        self._use_blessed = self.out_stream == sys.__stdout__ and self.in_stream == sys.__stdin__
        self._last_cursor_column: Optional[int] = None
        self._last_cursor_row: Optional[int] = None
        self.keep_last_line = keep_last_line
        self.extra_bytes_callback = extra_bytes_callback
        self.another_sigwinch = False
        self.in_get_cursor_diff = False

    def __enter__(self) -> 'CursorAwareWindow':
        self.cbreak = Cbreak(self.in_stream) if not self._use_blessed else self.t.cbreak()
        self.cbreak.__enter__()
        self.top_usable_row, _ = self.get_cursor_position()
        self._orig_top_usable_row = self.top_usable_row
        logger.debug('initial top_usable_row: %d' % self.top_usable_row)
        return super().__enter__()

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        if self.keep_last_line:
            self.write(self.t.move_down)
        self.write(self.t.move_x(0))
        self.write(self.t.clear_eos)
        self.write(self.t.clear_eol)
        self.cbreak.__exit__(type, value, traceback)
        super().__exit__(type, value, traceback)

    def get_cursor_position(self) -> Tuple[int, int]:
        """Returns the terminal (row, column) of the cursor

        0-indexed, like blessed cursor positions"""
        if self._use_blessed:
            return self.t.get_location()
        in_stream = self.in_stream
        query_cursor_position = '\x1b[6n'
        self.write(query_cursor_position)

        def retrying_read() -> str:
            while True:
                try:
                    c = in_stream.read(1)
                    if c == '':
                        raise ValueError("Stream should be blocking - shouldn't return ''. Returned %r so far", (resp,))
                    return c
                except OSError:
                    logger.info('stdin.read(1) that should never error just errored.')
                    continue
        resp = ''
        while True:
            c = retrying_read()
            resp += c
            m = re.search('(?P<extra>.*)(?P<CSI>\\x1b\\[|\\x9b)(?P<row>\\d+);(?P<column>\\d+)R', resp, re.DOTALL)
            if m:
                row = int(m.groupdict()['row'])
                col = int(m.groupdict()['column'])
                extra = m.groupdict()['extra']
                if extra:
                    if self.extra_bytes_callback is not None:
                        self.extra_bytes_callback(extra.encode(cast(TextIO, in_stream).encoding))
                    else:
                        raise ValueError('Bytes preceding cursor position query response thrown out:\n%r\nPass an extra_bytes_callback to CursorAwareWindow to prevent this' % (extra,))
                return (row - 1, col - 1)

    def get_cursor_vertical_diff(self) -> int:
        """Returns the how far down the cursor moved since last render.

        Note:
            If another get_cursor_vertical_diff call is already in progress,
            immediately returns zero. (This situation is likely if
            get_cursor_vertical_diff is called from a SIGWINCH signal
            handler, since sigwinches can happen in rapid succession and
            terminal emulators seem not to respond to cursor position
            queries before the next sigwinch occurs.)
        """
        if self.in_get_cursor_diff:
            self.another_sigwinch = True
            return 0
        cursor_dy = 0
        while True:
            self.in_get_cursor_diff = True
            self.another_sigwinch = False
            cursor_dy += self._get_cursor_vertical_diff_once()
            self.in_get_cursor_diff = False
            if not self.another_sigwinch:
                return cursor_dy

    def _get_cursor_vertical_diff_once(self) -> int:
        """Returns the how far down the cursor moved."""
        old_top_usable_row = self.top_usable_row
        row, col = self.get_cursor_position()
        if self._last_cursor_row is None:
            cursor_dy = 0
        else:
            cursor_dy = row - self._last_cursor_row
            logger.info('cursor moved %d lines down' % cursor_dy)
            while self.top_usable_row > -1 and cursor_dy > 0:
                self.top_usable_row += 1
                cursor_dy -= 1
            while self.top_usable_row > 1 and cursor_dy < 0:
                self.top_usable_row -= 1
                cursor_dy += 1
        logger.info('top usable row changed from %d to %d', old_top_usable_row, self.top_usable_row)
        logger.info('returning cursor dy of %d from curtsies' % cursor_dy)
        self._last_cursor_row = row
        return cursor_dy

    def render_to_terminal(self, array: Union[FSArray, Sequence[FmtStr]], cursor_pos: Tuple[int, int]=(0, 0)) -> int:
        """Renders array to terminal, returns the number of lines scrolled offscreen

        Returns:
            Number of times scrolled

        Args:
          array (FSArray): Grid of styled characters to be rendered.

            If array received is of width too small, render it anyway

            if array received is of width too large, render it anyway

            if array received is of height too small, render it anyway

            if array received is of height too large, render it, scroll down,
            and render the rest of it, then return how much we scrolled down

        """
        for_stdout = self.fmtstr_to_stdout_xform()
        if not self.hide_cursor:
            self.write(self.t.hide_cursor)
        height, width = (self.t.height, self.t.width)
        if height != self._last_rendered_height or width != self._last_rendered_width:
            self.on_terminal_size_change(height, width)
        current_lines_by_row: Dict[int, Optional[FmtStr]] = {}
        rows_for_use = list(range(self.top_usable_row, height))
        shared = min(len(array), len(rows_for_use))
        for row, line in zip(rows_for_use[:shared], array[:shared]):
            current_lines_by_row[row] = line
            if line == self._last_lines_by_row.get(row, None):
                continue
            self.write(self.t.move(row, 0))
            self.write(for_stdout(line))
            if len(line) < width:
                self.write(self.t.clear_eol)
        rest_of_lines = array[shared:]
        rest_of_rows = rows_for_use[shared:]
        for row in rest_of_rows:
            if self._last_lines_by_row and row not in self._last_lines_by_row:
                continue
            self.write(self.t.move(row, 0))
            self.write(self.t.clear_eol)
            self.write(self.t.clear_bol)
            current_lines_by_row[row] = None
        offscreen_scrolls = 0
        for line in rest_of_lines:
            self.scroll_down()
            if self.top_usable_row > 0:
                self.top_usable_row -= 1
            else:
                offscreen_scrolls += 1
            current_lines_by_row = {k - 1: v for k, v in current_lines_by_row.items()}
            logger.debug('new top_usable_row: %d' % self.top_usable_row)
            self.write(self.t.move(height - 1, 0))
            self.write(for_stdout(line))
            current_lines_by_row[height - 1] = line
        logger.debug('lines in last lines by row: %r' % self._last_lines_by_row.keys())
        logger.debug('lines in current lines by row: %r' % current_lines_by_row.keys())
        self._last_cursor_row = max(0, cursor_pos[0] - offscreen_scrolls + self.top_usable_row)
        self._last_cursor_column = cursor_pos[1]
        self.write(self.t.move(self._last_cursor_row, self._last_cursor_column))
        self._last_lines_by_row = current_lines_by_row
        if not self.hide_cursor:
            self.write(self.t.normal_cursor)
        return offscreen_scrolls