from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
class DAPGrouper(object):
    """
    Note: this is a helper class to group variables on the debug adapter protocol (DAP). For
    the xml protocol the type is just added to each variable and the UI can group/hide it as needed.
    """
    SCOPE_SPECIAL_VARS = 'special variables'
    SCOPE_PROTECTED_VARS = 'protected variables'
    SCOPE_FUNCTION_VARS = 'function variables'
    SCOPE_CLASS_VARS = 'class variables'
    SCOPES_SORTED = [SCOPE_SPECIAL_VARS, SCOPE_PROTECTED_VARS, SCOPE_FUNCTION_VARS, SCOPE_CLASS_VARS]
    __slots__ = ['variable_reference', 'scope', 'contents_debug_adapter_protocol']

    def __init__(self, scope):
        self.variable_reference = id(self)
        self.scope = scope
        self.contents_debug_adapter_protocol = []

    def get_contents_debug_adapter_protocol(self):
        return self.contents_debug_adapter_protocol[:]

    def __eq__(self, o):
        if isinstance(o, ScopeRequest):
            return self.variable_reference == o.variable_reference and self.scope == o.scope
        return False

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash((self.variable_reference, self.scope))

    def __repr__(self):
        return ''

    def __str__(self):
        return ''