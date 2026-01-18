import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _continue_batched_recheck(self, start_mark):
    if start_mark.get_buffer() != self._buffer:
        return
    start = self._buffer.get_iter_at_mark(start_mark)
    self._buffer.delete_mark(start_mark)
    if not self._enabled:
        return
    end = start.copy()
    end.forward_chars(_BATCH_SIZE_CHARS)
    self._iter_worker.forward_word_end(end)
    self.check_range(start, end, True)
    if not end.is_end():
        end.forward_char()
        start_mark = self._buffer.create_mark(None, end)
        GLib.idle_add(self._continue_batched_recheck, start_mark)