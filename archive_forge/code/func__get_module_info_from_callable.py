import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
def _get_module_info_from_callable(callable_):
    return _get_module_info(callable_.__globals__['__name__'])