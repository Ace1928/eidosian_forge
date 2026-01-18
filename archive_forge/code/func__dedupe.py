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
def _dedupe(_list):
    _set = {}
    _list.reverse()
    _list = [_set.setdefault(e, e) for e in _list if e not in _set]
    _list.reverse()
    return _list