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
class MenuEntryCache:
    """Class to cache Desktop Entries"""

    def __init__(self):
        self.cacheEntries = {}
        self.cacheEntries['legacy'] = []
        self.cache = {}

    def add_menu_entries(self, dirs, prefix='', legacy=False):
        for dir_ in dirs:
            if not dir_ in self.cacheEntries:
                self.cacheEntries[dir_] = []
                self.__addFiles(dir_, '', prefix, legacy)

    def __addFiles(self, dir_, subdir, prefix, legacy):
        for item in os.listdir(os.path.join(dir_, subdir)):
            if item.endswith('.desktop'):
                try:
                    menuentry = MenuEntry(os.path.join(subdir, item), dir_, prefix)
                except ParsingError:
                    continue
                self.cacheEntries[dir_].append(menuentry)
                if legacy:
                    self.cacheEntries['legacy'].append(menuentry)
            elif os.path.isdir(os.path.join(dir_, subdir, item)) and (not legacy):
                self.__addFiles(dir_, os.path.join(subdir, item), prefix, legacy)

    def get_menu_entries(self, dirs, legacy=True):
        entries = []
        ids = set()
        appdirs = dirs[:]
        if legacy:
            appdirs.append('legacy')
        key = ''.join(appdirs)
        try:
            return self.cache[key]
        except KeyError:
            pass
        for dir_ in appdirs:
            for menuentry in self.cacheEntries[dir_]:
                try:
                    if menuentry.DesktopFileID not in ids:
                        ids.add(menuentry.DesktopFileID)
                        entries.append(menuentry)
                    elif menuentry.getType() == MenuEntry.TYPE_SYSTEM:
                        idx = entries.index(menuentry)
                        entry = entries[idx]
                        if entry.getType() == MenuEntry.TYPE_USER:
                            entry.Original = menuentry
                except UnicodeDecodeError:
                    continue
        self.cache[key] = entries
        return entries