import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _after_text_insert(self, textbuffer, location, text, length):
    start = self._marks['insert-start'].iter
    self.check_range(start, location)
    self._marks['insert-end'].move(location)