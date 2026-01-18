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
def _ast_const(name):
    if sys.version_info >= (3, 4):
        name = ast.literal_eval(name)
        if sys.version_info >= (3, 8):
            return ast.Constant(name)
        else:
            return ast.NameConstant(name)
    else:
        return ast.Name(id=name, ctx=ast.Load())