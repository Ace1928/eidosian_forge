from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydevd_bundle.pydevconsole_code import InteractiveConsole
import os
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE
import traceback
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_save_locals
from _pydev_bundle.pydev_imports import Exec, _queue
import builtins as __builtin__
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
def activate_mpl_if_already_imported(interpreter):
    if interpreter.mpl_modules_for_patching:
        for module in list(interpreter.mpl_modules_for_patching):
            if module in sys.modules:
                activate_function = interpreter.mpl_modules_for_patching.pop(module)
                activate_function()