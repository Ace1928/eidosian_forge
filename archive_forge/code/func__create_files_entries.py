from weakref import ref
from time import time
from kivy.core.text import DEFAULT_FONT
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.utils import platform as core_platform
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import (
import collections.abc
from os import listdir
from os.path import (
from fnmatch import fnmatch
def _create_files_entries(self, *args):
    start = time()
    finished = False
    index = total = count = 1
    while time() - start < 0.05 or count < 10:
        try:
            index, total, item = next(self._gitems_gen)
            self._gitems.append(item)
            count += 1
        except StopIteration:
            finished = True
            break
        except TypeError:
            finished = True
            break
    if not finished:
        self._show_progress()
        self._progress.total = total
        self._progress.index = index
        return True
    self._items = items = self._gitems
    parent = self._gitems_parent
    if parent is None:
        self.dispatch('on_entries_cleared')
        for entry in items:
            self.dispatch('on_entry_added', entry, parent)
    else:
        parent.entries[:] = items
        for entry in items:
            self.dispatch('on_subentry_to_entry', entry, parent)
    self.files[:] = self._get_file_paths(items)
    self._hide_progress()
    self._gitems = None
    self._gitems_gen = None
    ev = self._create_files_entries_ev
    if ev is not None:
        ev.cancel()
    return False