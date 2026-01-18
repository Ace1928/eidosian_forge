import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def _remove_overridden_styles(styles_to_parse: List[str]) -> List[str]:
    """
    Utility function for align_text() / truncate_line() which filters a style list down
    to only those which would still be in effect if all were processed in order.

    This is mainly used to reduce how many style strings are stored in memory when
    building large multiline strings with ANSI styles. We only need to carry over
    styles from previous lines that are still in effect.

    :param styles_to_parse: list of styles to evaluate.
    :return: list of styles that are still in effect.
    """
    from . import ansi

    class StyleState:
        """Keeps track of what text styles are enabled"""

        def __init__(self) -> None:
            self.style_dict: Dict[int, str] = dict()
            self.reset_all: Optional[int] = None
            self.fg: Optional[int] = None
            self.bg: Optional[int] = None
            self.intensity: Optional[int] = None
            self.italic: Optional[int] = None
            self.overline: Optional[int] = None
            self.strikethrough: Optional[int] = None
            self.underline: Optional[int] = None
    style_state = StyleState()
    for index, style in enumerate(styles_to_parse):
        if style in (str(ansi.TextStyle.RESET_ALL), str(ansi.TextStyle.ALT_RESET_ALL)):
            style_state = StyleState()
            style_state.reset_all = index
        elif ansi.STD_FG_RE.match(style) or ansi.EIGHT_BIT_FG_RE.match(style) or ansi.RGB_FG_RE.match(style):
            if style_state.fg is not None:
                style_state.style_dict.pop(style_state.fg)
            style_state.fg = index
        elif ansi.STD_BG_RE.match(style) or ansi.EIGHT_BIT_BG_RE.match(style) or ansi.RGB_BG_RE.match(style):
            if style_state.bg is not None:
                style_state.style_dict.pop(style_state.bg)
            style_state.bg = index
        elif style in (str(ansi.TextStyle.INTENSITY_BOLD), str(ansi.TextStyle.INTENSITY_DIM), str(ansi.TextStyle.INTENSITY_NORMAL)):
            if style_state.intensity is not None:
                style_state.style_dict.pop(style_state.intensity)
            style_state.intensity = index
        elif style in (str(ansi.TextStyle.ITALIC_ENABLE), str(ansi.TextStyle.ITALIC_DISABLE)):
            if style_state.italic is not None:
                style_state.style_dict.pop(style_state.italic)
            style_state.italic = index
        elif style in (str(ansi.TextStyle.OVERLINE_ENABLE), str(ansi.TextStyle.OVERLINE_DISABLE)):
            if style_state.overline is not None:
                style_state.style_dict.pop(style_state.overline)
            style_state.overline = index
        elif style in (str(ansi.TextStyle.STRIKETHROUGH_ENABLE), str(ansi.TextStyle.STRIKETHROUGH_DISABLE)):
            if style_state.strikethrough is not None:
                style_state.style_dict.pop(style_state.strikethrough)
            style_state.strikethrough = index
        elif style in (str(ansi.TextStyle.UNDERLINE_ENABLE), str(ansi.TextStyle.UNDERLINE_DISABLE)):
            if style_state.underline is not None:
                style_state.style_dict.pop(style_state.underline)
            style_state.underline = index
        style_state.style_dict[index] = style
    return list(style_state.style_dict.values())