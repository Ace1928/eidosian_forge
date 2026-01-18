import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def _safe_import_hook(self, name, caller, fromlist, level=-1):
    if name in self.badmodules:
        self._add_badmodule(name, caller)
        return
    try:
        self.import_hook(name, caller, level=level)
    except ImportError as msg:
        self.msg(2, 'ImportError:', str(msg))
        self._add_badmodule(name, caller)
    except SyntaxError as msg:
        self.msg(2, 'SyntaxError:', str(msg))
        self._add_badmodule(name, caller)
    else:
        if fromlist:
            for sub in fromlist:
                fullname = name + '.' + sub
                if fullname in self.badmodules:
                    self._add_badmodule(fullname, caller)
                    continue
                try:
                    self.import_hook(name, caller, [sub], level=level)
                except ImportError as msg:
                    self.msg(2, 'ImportError:', str(msg))
                    self._add_badmodule(fullname, caller)