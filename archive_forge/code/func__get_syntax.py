import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pip._vendor.pygments.lexer import Lexer
from pip._vendor.pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pip._vendor.pygments.style import Style as PygmentsStyle
from pip._vendor.pygments.styles import get_style_by_name
from pip._vendor.pygments.token import (
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.rich.containers import Lines
from pip._vendor.rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
def _get_syntax(self, console: Console, options: ConsoleOptions) -> Iterable[Segment]:
    """
        Get the Segments for the Syntax object, excluding any vertical/horizontal padding
        """
    transparent_background = self._get_base_style().transparent_background
    code_width = (options.max_width - self._numbers_column_width - 1 if self.line_numbers else options.max_width) if self.code_width is None else self.code_width
    ends_on_nl, processed_code = self._process_code(self.code)
    text = self.highlight(processed_code, self.line_range)
    if not self.line_numbers and (not self.word_wrap) and (not self.line_range):
        if not ends_on_nl:
            text.remove_suffix('\n')
        style = self._get_base_style() + self._theme.get_style_for_token(Comment) + Style(dim=True) + self.background_style
        if self.indent_guides and (not options.ascii_only):
            text = text.with_indent_guides(self.tab_size, style=style)
            text.overflow = 'crop'
        if style.transparent_background:
            yield from console.render(text, options=options.update(width=code_width))
        else:
            syntax_lines = console.render_lines(text, options.update(width=code_width, height=None, justify='left'), style=self.background_style, pad=True, new_lines=True)
            for syntax_line in syntax_lines:
                yield from syntax_line
        return
    start_line, end_line = self.line_range or (None, None)
    line_offset = 0
    if start_line:
        line_offset = max(0, start_line - 1)
    lines: Union[List[Text], Lines] = text.split('\n', allow_blank=ends_on_nl)
    if self.line_range:
        if line_offset > len(lines):
            return
        lines = lines[line_offset:end_line]
    if self.indent_guides and (not options.ascii_only):
        style = self._get_base_style() + self._theme.get_style_for_token(Comment) + Style(dim=True) + self.background_style
        lines = Text('\n').join(lines).with_indent_guides(self.tab_size, style=style + Style(italic=False)).split('\n', allow_blank=True)
    numbers_column_width = self._numbers_column_width
    render_options = options.update(width=code_width)
    highlight_line = self.highlight_lines.__contains__
    _Segment = Segment
    new_line = _Segment('\n')
    line_pointer = '> ' if options.legacy_windows else '❱ '
    background_style, number_style, highlight_number_style = self._get_number_styles(console)
    for line_no, line in enumerate(lines, self.start_line + line_offset):
        if self.word_wrap:
            wrapped_lines = console.render_lines(line, render_options.update(height=None, justify='left'), style=background_style, pad=not transparent_background)
        else:
            segments = list(line.render(console, end=''))
            if options.no_wrap:
                wrapped_lines = [segments]
            else:
                wrapped_lines = [_Segment.adjust_line_length(segments, render_options.max_width, style=background_style, pad=not transparent_background)]
        if self.line_numbers:
            wrapped_line_left_pad = _Segment(' ' * numbers_column_width + ' ', background_style)
            for first, wrapped_line in loop_first(wrapped_lines):
                if first:
                    line_column = str(line_no).rjust(numbers_column_width - 2) + ' '
                    if highlight_line(line_no):
                        yield _Segment(line_pointer, Style(color='red'))
                        yield _Segment(line_column, highlight_number_style)
                    else:
                        yield _Segment('  ', highlight_number_style)
                        yield _Segment(line_column, number_style)
                else:
                    yield wrapped_line_left_pad
                yield from wrapped_line
                yield new_line
        else:
            for wrapped_line in wrapped_lines:
                yield from wrapped_line
                yield new_line