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
class FileChooserLayout(FloatLayout):
    """Base class for file chooser layouts.

    .. versionadded:: 1.9.0
    """
    VIEWNAME = 'undefined'
    __events__ = ('on_entry_added', 'on_entries_cleared', 'on_subentry_to_entry', 'on_remove_subentry', 'on_submit')
    controller = ObjectProperty()
    '\n    Reference to the controller handling this layout.\n\n    :class:`~kivy.properties.ObjectProperty`\n    '

    def on_entry_added(self, node, parent=None):
        pass

    def on_entries_cleared(self):
        pass

    def on_subentry_to_entry(self, subentry, entry):
        pass

    def on_remove_subentry(self, subentry, entry):
        pass

    def on_submit(self, selected, touch=None):
        pass