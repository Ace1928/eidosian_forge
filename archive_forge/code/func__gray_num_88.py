from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _gray_num_88(gnum: int) -> int:
    """Return ths color number for gray number gnum.

    Color cube black and white are returned for 0 and 9 respectively
    since those values aren't included in the gray scale.

    """
    gnum -= 1
    if gnum < 0:
        return _CUBE_BLACK
    if gnum >= _GRAY_SIZE_88:
        return _CUBE_WHITE_88
    return _GRAY_START_88 + gnum