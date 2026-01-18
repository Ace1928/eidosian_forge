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
def _compile_text(template, text, filename):
    identifier = template.module_id
    source, lexer = _compile(template, text, filename, generate_magic_comment=False)
    cid = identifier
    module = types.ModuleType(cid)
    code = compile(source, cid, 'exec')
    exec(code, module.__dict__, module.__dict__)
    return (source, module)