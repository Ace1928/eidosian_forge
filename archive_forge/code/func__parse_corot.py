from __future__ import annotations
from string import Formatter
from typing import Generator
from prompt_toolkit.output.vt100 import BG_ANSI_COLORS, FG_ANSI_COLORS
from prompt_toolkit.output.vt100 import _256_colors as _256_colors_table
from .base import StyleAndTextTuples
def _parse_corot(self) -> Generator[None, str, None]:
    """
        Coroutine that parses the ANSI escape sequences.
        """
    style = ''
    formatted_text = self._formatted_text
    while True:
        csi = False
        c = (yield)
        if c == '\x01':
            escaped_text = ''
            while c != '\x02':
                c = (yield)
                if c == '\x02':
                    formatted_text.append(('[ZeroWidthEscape]', escaped_text))
                    c = (yield)
                    break
                else:
                    escaped_text += c
        if c == '\x1b':
            square_bracket = (yield)
            if square_bracket == '[':
                csi = True
            else:
                continue
        elif c == '\x9b':
            csi = True
        if csi:
            current = ''
            params = []
            while True:
                char = (yield)
                if char.isdigit():
                    current += char
                else:
                    params.append(min(int(current or 0), 9999))
                    if char == ';':
                        current = ''
                    elif char == 'm':
                        self._select_graphic_rendition(params)
                        style = self._create_style_string()
                        break
                    elif char == 'C':
                        for i in range(params[0]):
                            formatted_text.append((style, ' '))
                        break
                    else:
                        break
        else:
            formatted_text.append((style, c))