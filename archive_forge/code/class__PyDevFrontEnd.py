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
class _PyDevFrontEnd:
    version = release.__version__

    def __init__(self):
        if hasattr(PyDevTerminalInteractiveShell, '_instance') and PyDevTerminalInteractiveShell._instance is not None:
            self.ipython = PyDevTerminalInteractiveShell._instance
        else:
            self.ipython = PyDevTerminalInteractiveShell.instance()
        self._curr_exec_line = 0
        self._curr_exec_lines = []

    def show_banner(self):
        self.ipython.show_banner()

    def update(self, globals, locals):
        ns = self.ipython.user_ns
        for key, value in list(ns.items()):
            if key not in locals:
                locals[key] = value
        self.ipython.user_global_ns.clear()
        self.ipython.user_global_ns.update(globals)
        self.ipython.user_ns = locals
        if hasattr(self.ipython, 'history_manager') and hasattr(self.ipython.history_manager, 'save_thread'):
            self.ipython.history_manager.save_thread.pydev_do_not_trace = True

    def complete(self, string):
        try:
            if string:
                return self.ipython.complete(None, line=string, cursor_pos=string.__len__())
            else:
                return self.ipython.complete(string, string, 0)
        except:
            pass

    def is_complete(self, string):
        if string in ('', '\n'):
            return True
        else:
            try:
                clean_string = string.rstrip('\n')
                if not clean_string.endswith('\\'):
                    clean_string += '\n\n'
                is_complete = codeop.compile_command(clean_string, '<string>', 'exec')
            except Exception:
                is_complete = True
            return is_complete

    def getCompletions(self, text, act_tok):
        try:
            TYPE_IPYTHON = '11'
            TYPE_IPYTHON_MAGIC = '12'
            _line, ipython_completions = self.complete(text)
            from _pydev_bundle._pydev_completer import Completer
            completer = Completer(self.get_namespace(), None)
            ret = completer.complete(act_tok)
            append = ret.append
            ip = self.ipython
            pydev_completions = set([f[0] for f in ret])
            for ipython_completion in ipython_completions:
                if ipython_completion not in pydev_completions:
                    pydev_completions.add(ipython_completion)
                    inf = ip.object_inspect(ipython_completion)
                    if inf['type_name'] == 'Magic function':
                        pydev_type = TYPE_IPYTHON_MAGIC
                    else:
                        pydev_type = TYPE_IPYTHON
                    pydev_doc = inf['docstring']
                    if pydev_doc is None:
                        pydev_doc = ''
                    append((ipython_completion, pydev_doc, '', pydev_type))
            return ret
        except:
            import traceback
            traceback.print_exc()
            return []

    def get_namespace(self):
        return self.ipython.user_ns

    def clear_buffer(self):
        del self._curr_exec_lines[:]

    def add_exec(self, line):
        if self._curr_exec_lines:
            self._curr_exec_lines.append(line)
            buf = '\n'.join(self._curr_exec_lines)
            if self.is_complete(buf):
                self._curr_exec_line += 1
                self.ipython.run_cell(buf)
                del self._curr_exec_lines[:]
                return False
            return True
        elif not self.is_complete(line):
            self._curr_exec_lines.append(line)
            return True
        else:
            self._curr_exec_line += 1
            self.ipython.run_cell(line, store_history=True)
            return False

    def is_automagic(self):
        return self.ipython.automagic

    def get_greeting_msg(self):
        return 'PyDev console: using IPython %s\n' % self.version