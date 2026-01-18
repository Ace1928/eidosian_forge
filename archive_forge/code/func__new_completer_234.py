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
def _new_completer_234(self):
    completer = PyDevIPCompleter(shell=self, namespace=self.user_ns, global_namespace=self.user_global_ns, use_readline=self.has_readline, parent=self)
    return completer