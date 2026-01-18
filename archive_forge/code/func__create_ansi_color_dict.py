from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def _create_ansi_color_dict(color_cls):
    """ Create a table that maps the 16 named ansi colors to their Windows code. """
    return {'ansidefault': color_cls.BLACK, 'ansiblack': color_cls.BLACK, 'ansidarkgray': color_cls.BLACK | color_cls.INTENSITY, 'ansilightgray': color_cls.GRAY, 'ansiwhite': color_cls.GRAY | color_cls.INTENSITY, 'ansidarkred': color_cls.RED, 'ansidarkgreen': color_cls.GREEN, 'ansibrown': color_cls.YELLOW, 'ansidarkblue': color_cls.BLUE, 'ansipurple': color_cls.MAGENTA, 'ansiteal': color_cls.CYAN, 'ansired': color_cls.RED | color_cls.INTENSITY, 'ansigreen': color_cls.GREEN | color_cls.INTENSITY, 'ansiyellow': color_cls.YELLOW | color_cls.INTENSITY, 'ansiblue': color_cls.BLUE | color_cls.INTENSITY, 'ansifuchsia': color_cls.MAGENTA | color_cls.INTENSITY, 'ansiturquoise': color_cls.CYAN | color_cls.INTENSITY}