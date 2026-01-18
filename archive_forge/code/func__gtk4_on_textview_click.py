import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _gtk4_on_textview_click(self, click, n_press, x, y) -> None:
    if n_press != 1 or click.get_current_button() != 3:
        return
    self._move_mark_for_input(x, y)
    self.populate_menu(self._spelling_menu)