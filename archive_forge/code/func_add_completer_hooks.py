from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
def add_completer_hooks(self):
    from IPython.core.completerlib import module_completer, magic_run_completer, cd_completer
    try:
        from IPython.core.completerlib import reset_completer
    except ImportError:
        reset_completer = None
    self.configurables.append(self.Completer)
    sdisp = self.strdispatchers.get('complete_command', StrDispatch())
    self.strdispatchers['complete_command'] = sdisp
    self.Completer.custom_completers = sdisp
    self.set_hook('complete_command', module_completer, str_key='import')
    self.set_hook('complete_command', module_completer, str_key='from')
    self.set_hook('complete_command', magic_run_completer, str_key='%run')
    self.set_hook('complete_command', cd_completer, str_key='%cd')
    if reset_completer:
        self.set_hook('complete_command', reset_completer, str_key='%reset')