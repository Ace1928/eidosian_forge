import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _move_mark_for_input(self, input_x, input_y):
    if self._deferred_check:
        self._check_deferred_range(True)
    x, y = self._view.window_to_buffer_coords(2, int(input_x), int(input_y))
    iter = self._view.get_iter_at_location(x, y)
    if isinstance(iter, tuple):
        iter = iter[1]
    self.move_click_mark(iter)