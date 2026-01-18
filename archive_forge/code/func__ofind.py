import abc
import ast
import atexit
import bdb
import builtins as builtin_mod
import functools
import inspect
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import warnings
from ast import stmt
from io import open as io_open
from logging import error
from pathlib import Path
from typing import Callable
from typing import List as ListType, Dict as DictType, Any as AnyType
from typing import Optional, Sequence, Tuple
from warnings import warn
from tempfile import TemporaryDirectory
from traitlets import (
from traitlets.config.configurable import SingletonConfigurable
from traitlets.utils.importstring import import_item
import IPython.core.hooks
from IPython.core import magic, oinspect, page, prefilter, ultratb
from IPython.core.alias import Alias, AliasManager
from IPython.core.autocall import ExitAutocall
from IPython.core.builtin_trap import BuiltinTrap
from IPython.core.compilerop import CachingCompiler
from IPython.core.debugger import InterruptiblePdb
from IPython.core.display_trap import DisplayTrap
from IPython.core.displayhook import DisplayHook
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import InputRejected, UsageError
from IPython.core.events import EventManager, available_events
from IPython.core.extensions import ExtensionManager
from IPython.core.formatters import DisplayFormatter
from IPython.core.history import HistoryManager
from IPython.core.inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from IPython.core.logger import Logger
from IPython.core.macro import Macro
from IPython.core.payload import PayloadManager
from IPython.core.prefilter import PrefilterManager
from IPython.core.profiledir import ProfileDir
from IPython.core.usage import default_banner
from IPython.display import display
from IPython.paths import get_ipython_dir
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, io, openpy, py3compat
from IPython.utils.decorators import undoc
from IPython.utils.io import ask_yes_no
from IPython.utils.ipstruct import Struct
from IPython.utils.path import ensure_dir_exists, get_home_dir, get_py_filename
from IPython.utils.process import getoutput, system
from IPython.utils.strdispatch import StrDispatch
from IPython.utils.syspathcontext import prepended_to_syspath
from IPython.utils.text import DollarFormatter, LSString, SList, format_screen
from IPython.core.oinspect import OInfo
from ast import Module
from .async_helpers import (
def _ofind(self, oname: str, namespaces: Optional[Sequence[Tuple[str, AnyType]]]=None) -> OInfo:
    """Find an object in the available namespaces.


        Returns
        -------
        OInfo with fields:
          - ismagic
          - isalias
          - found
          - obj
          - namespac
          - parent

        Has special code to detect magic functions.
        """
    oname = oname.strip()
    parts_ok, parts = self._find_parts(oname)
    if not oname.startswith(ESC_MAGIC) and (not oname.startswith(ESC_MAGIC2)) and (not parts_ok):
        return OInfo(ismagic=False, isalias=False, found=False, obj=None, namespace=None, parent=None)
    if namespaces is None:
        namespaces = [('Interactive', self.user_ns), ('Interactive (global)', self.user_global_ns), ('Python builtin', builtin_mod.__dict__)]
    ismagic = False
    isalias = False
    found = False
    ospace = None
    parent = None
    obj = None
    oname_parts = parts
    oname_head, oname_rest = (oname_parts[0], oname_parts[1:])
    for nsname, ns in namespaces:
        try:
            obj = ns[oname_head]
        except KeyError:
            continue
        else:
            for idx, part in enumerate(oname_rest):
                try:
                    parent = obj
                    if idx == len(oname_rest) - 1:
                        obj = self._getattr_property(obj, part)
                    elif is_integer_string(part):
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                except:
                    break
            else:
                found = True
                ospace = nsname
                break
    if not found:
        obj = None
        if oname.startswith(ESC_MAGIC2):
            oname = oname.lstrip(ESC_MAGIC2)
            obj = self.find_cell_magic(oname)
        elif oname.startswith(ESC_MAGIC):
            oname = oname.lstrip(ESC_MAGIC)
            obj = self.find_line_magic(oname)
        else:
            obj = self.find_line_magic(oname)
            if obj is None:
                obj = self.find_cell_magic(oname)
        if obj is not None:
            found = True
            ospace = 'IPython internal'
            ismagic = True
            isalias = isinstance(obj, Alias)
    if not found and oname_head in ["''", '""', '[]', '{}', '()']:
        obj = eval(oname_head)
        found = True
        ospace = 'Interactive'
    return OInfo(obj=obj, found=found, parent=parent, ismagic=ismagic, isalias=isalias, namespace=ospace)