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
class FileSystemLocal(FileSystemAbstract):
    """Implementation of :class:`FileSystemAbstract` for local files.

    .. versionadded:: 1.8.0
    """

    def listdir(self, fn):
        return listdir(fn)

    def getsize(self, fn):
        return getsize(fn)

    def is_hidden(self, fn):
        if platform == 'win':
            if not _have_win32file:
                return False
            try:
                return GetFileAttributesExW(fn)[0] & FILE_ATTRIBUTE_HIDDEN
            except error:
                Logger.exception('unable to access to <%s>' % fn)
                return True
        return basename(fn).startswith('.')

    def is_dir(self, fn):
        return isdir(fn)