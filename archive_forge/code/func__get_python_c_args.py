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
def _get_python_c_args(host, port, code, args, setup):
    setup = _get_setup_updated_with_protocol_and_ppid(setup)
    setup_repr = setup if setup is None else sorted_dict_repr(setup)
    future_imports = ''
    if '__future__' in code:
        future_imports, code = _separate_future_imports(code)
    return "%simport sys; sys.path.insert(0, r'%s'); import pydevd; pydevd.config(%r, %r); pydevd.settrace(host=%r, port=%s, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=%r, client_access_token=%r, __setup_holder__=%s); %s" % (future_imports, pydev_src_dir, pydevd_constants.get_protocol(), PydevdCustomization.DEBUG_MODE, host, port, setup.get('access-token'), setup.get('client-access-token'), setup_repr, code)