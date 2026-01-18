import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def _parse_source(source_text, filename='<unknown>'):
    """Get object to lineno mappings from given source_text"""
    import ast
    cls_to_lineno = {}
    str_to_lineno = {}
    for node in ast.walk(ast.parse(source_text, filename)):
        if isinstance(node, ast.ClassDef):
            cls_to_lineno[node.name] = node.lineno
        elif isinstance(node, ast.Str):
            str_to_lineno[node.s] = node.lineno - (0 if sys.version_info >= (3, 8) else node.s.count('\n'))
    return (cls_to_lineno, str_to_lineno)