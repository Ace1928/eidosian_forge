from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
class MenuEntry:
    """Wrapper for 'Menu Style' Desktop Entries"""
    TYPE_USER = 'User'
    TYPE_SYSTEM = 'System'
    TYPE_BOTH = 'Both'

    def __init__(self, filename, dir='', prefix=''):
        self.DesktopEntry = DesktopEntry(os.path.join(dir, filename))
        self.setAttributes(filename, dir, prefix)
        self.Show = True
        self.Original = None
        self.Parents = []
        self.Allocated = False
        self.Add = False
        self.MatchedInclude = False
        self.Categories = self.DesktopEntry.getCategories()

    def save(self):
        """Save any changes to the desktop entry."""
        if self.DesktopEntry.tainted:
            self.DesktopEntry.write()

    def getDir(self):
        """Return the directory containing the desktop entry file."""
        return self.DesktopEntry.filename.replace(self.Filename, '')

    def getType(self):
        """Return the type of MenuEntry, System/User/Both"""
        if not xdg.Config.root_mode:
            if self.Original:
                return self.TYPE_BOTH
            elif xdg_data_dirs[0] in self.DesktopEntry.filename:
                return self.TYPE_USER
            else:
                return self.TYPE_SYSTEM
        else:
            return self.TYPE_USER

    def setAttributes(self, filename, dir='', prefix=''):
        self.Filename = filename
        self.Prefix = prefix
        self.DesktopFileID = os.path.join(prefix, filename).replace('/', '-')
        if not os.path.isabs(self.DesktopEntry.filename):
            self.__setFilename()

    def updateAttributes(self):
        if self.getType() == self.TYPE_SYSTEM:
            self.Original = MenuEntry(self.Filename, self.getDir(), self.Prefix)
            self.__setFilename()

    def __setFilename(self):
        if not xdg.Config.root_mode:
            path = xdg_data_dirs[0]
        else:
            path = xdg_data_dirs[1]
        if self.DesktopEntry.getType() == 'Application':
            dir_ = os.path.join(path, 'applications')
        else:
            dir_ = os.path.join(path, 'desktop-directories')
        self.DesktopEntry.filename = os.path.join(dir_, self.Filename)

    def __cmp__(self, other):
        return locale.strcoll(self.DesktopEntry.getName(), other.DesktopEntry.getName())

    def _key(self):
        """Key function for locale-aware sorting."""
        return _strxfrm(self.DesktopEntry.getName())

    def __lt__(self, other):
        try:
            other = other._key()
        except AttributeError:
            pass
        return self._key() < other

    def __eq__(self, other):
        if self.DesktopFileID == str(other):
            return True
        else:
            return False

    def __repr__(self):
        return self.DesktopFileID