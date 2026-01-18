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