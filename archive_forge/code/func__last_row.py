from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _last_row(self, row: list[tuple[object, Literal['0', 'U'] | None, bytes]]) -> tuple[list[tuple[object, Literal['0', 'U'] | None, bytes]], int, tuple[object, Literal['0', 'U'] | None, bytes]]:
    """On the last row we need to slide the bottom right character
        into place. Calculate the new line, attr and an insert sequence
        to do that.

        eg. last row:
        XXXXXXXXXXXXXXXXXXXXYZ

        Y will be drawn after Z, shifting Z into position.
        """
    new_row = row[:-1]
    z_attr, z_cs, last_text = row[-1]
    last_cols = str_util.calc_width(last_text, 0, len(last_text))
    last_offs, z_col = str_util.calc_text_pos(last_text, 0, len(last_text), last_cols - 1)
    if last_offs == 0:
        z_text = last_text
        del new_row[-1]
        y_attr, y_cs, nlast_text = row[-2]
        nlast_cols = str_util.calc_width(nlast_text, 0, len(nlast_text))
        z_col += nlast_cols
        nlast_offs, y_col = str_util.calc_text_pos(nlast_text, 0, len(nlast_text), nlast_cols - 1)
        y_text = nlast_text[nlast_offs:]
        if nlast_offs:
            new_row.append((y_attr, y_cs, nlast_text[:nlast_offs]))
    else:
        z_text = last_text[last_offs:]
        y_attr, y_cs = (z_attr, z_cs)
        nlast_cols = str_util.calc_width(last_text, 0, last_offs)
        nlast_offs, y_col = str_util.calc_text_pos(last_text, 0, last_offs, nlast_cols - 1)
        y_text = last_text[nlast_offs:last_offs]
        if nlast_offs:
            new_row.append((y_attr, y_cs, last_text[:nlast_offs]))
    new_row.append((z_attr, z_cs, z_text))
    return (new_row, z_col - y_col, (y_attr, y_cs, y_text))