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
def _show_progress(self):
    if self._progress:
        return
    cls = self.progress_cls
    if isinstance(cls, string_types):
        cls = Factory.get(cls)
    self._progress = cls(path=self.path)
    self._progress.value = 0
    self.add_widget(self._progress)