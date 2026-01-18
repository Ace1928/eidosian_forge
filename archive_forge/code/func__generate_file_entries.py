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
def _generate_file_entries(self, *args, **kwargs):
    is_root = False
    path = kwargs.get('path', self.path)
    have_parent = kwargs.get('parent', None) is not None
    if self.rootpath:
        rootpath = realpath(self.rootpath)
        path = realpath(path)
        if not path.startswith(rootpath):
            self.path = rootpath
            return
        elif path == rootpath:
            is_root = True
    elif platform == 'win':
        is_root = splitdrive(path)[1] in (sep, altsep)
    elif platform in ('macosx', 'linux', 'android', 'ios'):
        is_root = normpath(expanduser(path)) == sep
    else:
        Logger.warning('Filechooser: Unsupported OS: %r' % platform)
    if not is_root and (not have_parent):
        back = '..' + sep
        if platform == 'win':
            new_path = path[:path.rfind(sep)]
            if sep not in new_path:
                new_path += sep
            pardir = self._create_entry_widget(dict(name=back, size='', path=new_path, controller=ref(self), isdir=True, parent=None, sep=sep, get_nice_size=lambda: ''))
        else:
            pardir = self._create_entry_widget(dict(name=back, size='', path=back, controller=ref(self), isdir=True, parent=None, sep=sep, get_nice_size=lambda: ''))
        yield (0, 1, pardir)
    try:
        for index, total, item in self._add_files(path):
            yield (index, total, item)
    except OSError:
        Logger.exception('Unable to open directory <%s>' % self.path)
        self.files[:] = []