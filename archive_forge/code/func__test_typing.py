from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def _test_typing(code, inject=False):
    sys.path.insert(0, TOOLS_DIR)
    try:
        import jedityper
    finally:
        sys.path.remove(TOOLS_DIR)
    lines = []
    with _tempfile(code) as f:
        types = jedityper.analyse(f.name)
        if inject:
            lines = jedityper.inject_types(f.name, types)
    return (types, lines)