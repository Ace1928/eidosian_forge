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
def _check_file_path(value, filename, type):
    path = os.path.dirname(filename)
    if not os.path.isabs(value):
        value = os.path.join(path, value)
    value = os.path.abspath(value)
    if not os.path.exists(value):
        return False
    if type == TYPE_DIR and os.path.isdir(value):
        return value
    if type == TYPE_FILE and os.path.isfile(value):
        return value
    return False