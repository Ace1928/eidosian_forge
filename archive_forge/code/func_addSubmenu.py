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
def addSubmenu(self, newmenu):
    for submenu in self.Submenus:
        if submenu == newmenu:
            submenu += newmenu
            break
    else:
        self.Submenus.append(newmenu)
        newmenu.Parent = self
        newmenu.Depth = self.Depth + 1