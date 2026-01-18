import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _check_deferred_range(self, force_all):
    start = self._marks['insert-start'].iter
    end = self._marks['insert-end'].iter
    self.check_range(start, end, force_all)