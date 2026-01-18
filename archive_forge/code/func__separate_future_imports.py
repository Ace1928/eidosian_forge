import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def _separate_future_imports(code):
    """
    :param code:
        The code from where we want to get the __future__ imports (note that it's possible that
        there's no such entry).

    :return tuple(str, str):
        The return is a tuple(future_import, code).

        If the future import is not available a return such as ('', code) is given, otherwise, the
        future import will end with a ';' (so that it can be put right before the pydevd attach
        code).
    """
    try:
        node = ast.parse(code, '<string>', 'exec')
        visitor = _LastFutureImportFinder()
        visitor.visit(node)
        if visitor.last_future_import_found is None:
            return ('', code)
        node = visitor.last_future_import_found
        offset = -1
        if hasattr(node, 'end_lineno') and hasattr(node, 'end_col_offset'):
            line, col = (node.end_lineno, node.end_col_offset)
            offset = _get_offset_from_line_col(code, line - 1, col)
        else:
            line, col = (node.lineno, node.col_offset)
            offset = _get_offset_from_line_col(code, line - 1, col)
            if offset >= 0 and node.names:
                from_future_import_name = node.names[-1].name
                i = code.find(from_future_import_name, offset)
                if i < 0:
                    offset = -1
                else:
                    offset = i + len(from_future_import_name)
        if offset >= 0:
            for i in range(offset, len(code)):
                if code[i] in (' ', '\t', ';', ')', '\n'):
                    offset += 1
                else:
                    break
            future_import = code[:offset]
            code_remainder = code[offset:]
            while future_import.endswith('\n'):
                future_import = future_import[:-1]
                code_remainder = '\n' + code_remainder
            if not future_import.endswith(';'):
                future_import += ';'
            return (future_import, code_remainder)
        pydev_log.info('Unable to find line %s in code:\n%r', line, code)
        return ('', code)
    except:
        pydev_log.exception('Error getting from __future__ imports from: %r', code)
        return ('', code)