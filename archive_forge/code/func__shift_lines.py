import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def _shift_lines(self, direction, rows=None, old_cursor=None, from_undo=False):
    if self._selection_callback:
        if from_undo:
            self._selection_callback.cancel()
        else:
            return
    lines = self._lines
    flags = list(reversed(self._lines_flags))
    labels = self._lines_labels
    rects = self._lines_rects
    orig_cursor = self.cursor
    sel = None
    if old_cursor is not None:
        self.cursor = old_cursor
    if not rows:
        sindex = self.selection_from
        eindex = self.selection_to
        if (sindex or eindex) and sindex != eindex:
            sindex, eindex = tuple(sorted((sindex, eindex)))
            sindex, eindex = self._expand_range(sindex, eindex)
        else:
            sindex, eindex = self._expand_range(self.cursor_index())
        srow = self.get_cursor_from_index(sindex)[1]
        erow = self.get_cursor_from_index(eindex)[1]
        sel = (sindex, eindex)
        if direction < 0 and srow > 0:
            psrow, perow = self._expand_rows(srow - 1)
            rows = ((srow, erow), (psrow, perow))
        elif direction > 0 and erow < len(lines) - 1:
            psrow, perow = self._expand_rows(erow)
            rows = ((srow, erow), (psrow, perow))
    else:
        (srow, erow), (psrow, perow) = rows
        if direction < 0:
            m1srow, m1erow = (psrow, perow)
            m2srow, m2erow = (srow, erow)
            cdiff = psrow - perow
            xdiff = srow - erow
        else:
            m1srow, m1erow = (srow, erow)
            m2srow, m2erow = (psrow, perow)
            cdiff = perow - psrow
            xdiff = erow - srow
        self._lines_flags = list(reversed(chain(flags[:m1srow], flags[m2srow:m2erow], flags[m1srow:m1erow], flags[m2erow:])))
        self._lines[:] = lines[:m1srow] + lines[m2srow:m2erow] + lines[m1srow:m1erow] + lines[m2erow:]
        self._lines_labels = labels[:m1srow] + labels[m2srow:m2erow] + labels[m1srow:m1erow] + labels[m2erow:]
        self._lines_rects = rects[:m1srow] + rects[m2srow:m2erow] + rects[m1srow:m1erow] + rects[m2erow:]
        self._trigger_update_graphics()
        csrow = srow + cdiff
        cerow = erow + cdiff
        sel = (self.cursor_index((0, csrow)), self.cursor_index((0, cerow)))
        self.cursor = (self.cursor_col, self.cursor_row + cdiff)
        if not from_undo:
            undo_rows = ((srow + cdiff, erow + cdiff), (psrow - xdiff, perow - xdiff))
            self._undo.append({'undo_command': ('shiftln', direction * -1, undo_rows, self.cursor), 'redo_command': ('shiftln', direction, rows, orig_cursor)})
            self._redo = []
    if sel:

        def cb(dt):
            self.select_text(*sel)
            self._selection_callback = None
        self._selection_callback = Clock.schedule_once(cb)