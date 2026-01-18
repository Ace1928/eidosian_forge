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
class _LastFutureImportFinder(ast.NodeVisitor):

    def __init__(self):
        self.last_future_import_found = None

    def visit_ImportFrom(self, node):
        if node.module == '__future__':
            self.last_future_import_found = node