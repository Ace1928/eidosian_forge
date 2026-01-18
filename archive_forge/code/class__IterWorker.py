import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
class _IterWorker:

    def __init__(self, extra_word_chars):
        self._extra_word_chars = extra_word_chars

    def is_extra_word_char(self, loc):
        char = loc.get_char()
        return char != '' and char in self._extra_word_chars

    def inside_word(self, loc):
        if loc.inside_word():
            return True
        elif self.starts_word(loc):
            return True
        elif loc.ends_word() and (not self.ends_word(loc)):
            return True
        else:
            return False

    def starts_word(self, loc):
        if loc.starts_word():
            if loc.is_start():
                return True
            else:
                tmp = loc.copy()
                tmp.backward_char()
                return not self.is_extra_word_char(tmp)
        else:
            return False

    def ends_word(self, loc):
        if loc.ends_word():
            if loc.is_end():
                return True
            else:
                tmp = loc.copy()
                tmp.forward_char()
                return not self.is_extra_word_char(tmp)
        else:
            return False

    def forward_word_end(self, loc):

        def move_through_extra_chars():
            moved = False
            while self.is_extra_word_char(loc):
                if not loc.forward_char():
                    break
                moved = True
            return moved
        tmp = loc.copy()
        tmp.backward_char()
        loc.forward_word_end()
        while move_through_extra_chars():
            if loc.is_end() or not loc.inside_word() or (not loc.forward_word_end()):
                break

    def backward_word_start(self, loc):

        def move_through_extra_chars():
            tmp = loc.copy()
            tmp.backward_char()
            moved = False
            while self.is_extra_word_char(tmp):
                moved = True
                loc.assign(tmp)
                if not tmp.backward_char():
                    break
            return moved
        loc.backward_word_start()
        while move_through_extra_chars():
            tmp = loc.copy()
            tmp.backward_char()
            if loc.is_start() or not tmp.inside_word() or (not loc.backward_word_start()):
                break

    def sync_extra_chars(self, obj, value):
        self._extra_word_chars = obj.extra_chars