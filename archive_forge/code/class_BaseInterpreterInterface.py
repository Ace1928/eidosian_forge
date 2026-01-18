import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
class BaseInterpreterInterface:

    def __init__(self, mainThread, connect_status_queue=None):
        self.mainThread = mainThread
        self.interruptable = False
        self.exec_queue = _queue.Queue(0)
        self.buffer = None
        self.banner_shown = False
        self.connect_status_queue = connect_status_queue
        self.mpl_modules_for_patching = {}
        self.init_mpl_modules_for_patching()

    def build_banner(self):
        return 'print({0})\n'.format(repr(self.get_greeting_msg()))

    def get_greeting_msg(self):
        return 'PyDev console: starting.\n'

    def init_mpl_modules_for_patching(self):
        from pydev_ipython.matplotlibtools import activate_matplotlib, activate_pylab, activate_pyplot
        self.mpl_modules_for_patching = {'matplotlib': lambda: activate_matplotlib(self.enableGui), 'matplotlib.pyplot': activate_pyplot, 'pylab': activate_pylab}

    def need_more_for_code(self, source):
        if source.endswith('\\'):
            return True
        if hasattr(self.interpreter, 'is_complete'):
            return not self.interpreter.is_complete(source)
        try:
            symbol = 'single'
            code = self.interpreter.compile(source, '<input>', symbol)
        except (OverflowError, SyntaxError, ValueError):
            return False
        if code is None:
            return True
        return False

    def need_more(self, code_fragment):
        if self.buffer is None:
            self.buffer = code_fragment
        else:
            self.buffer.append(code_fragment)
        return self.need_more_for_code(self.buffer.text)

    def create_std_in(self, debugger=None, original_std_in=None):
        if debugger is None:
            return StdIn(self, self.host, self.client_port, original_stdin=original_std_in)
        else:
            return DebugConsoleStdIn(py_db=debugger, original_stdin=original_std_in)

    def add_exec(self, code_fragment, debugger=None):
        sys.excepthook = sys.__excepthook__
        try:
            original_in = sys.stdin
            try:
                help = None
                if 'pydoc' in sys.modules:
                    pydoc = sys.modules['pydoc']
                    if hasattr(pydoc, 'help'):
                        help = pydoc.help
                        if not hasattr(help, 'input'):
                            help = None
            except:
                pass
            more = False
            try:
                sys.stdin = self.create_std_in(debugger, original_in)
                try:
                    if help is not None:
                        try:
                            try:
                                help.input = sys.stdin
                            except AttributeError:
                                help._input = sys.stdin
                        except:
                            help = None
                            if not self._input_error_printed:
                                self._input_error_printed = True
                                sys.stderr.write('\nError when trying to update pydoc.help.input\n')
                                sys.stderr.write('(help() may not work -- please report this as a bug in the pydev bugtracker).\n\n')
                                traceback.print_exc()
                    try:
                        self.start_exec()
                        if hasattr(self, 'debugger'):
                            self.debugger.enable_tracing()
                        more = self.do_add_exec(code_fragment)
                        if hasattr(self, 'debugger'):
                            self.debugger.disable_tracing()
                        self.finish_exec(more)
                    finally:
                        if help is not None:
                            try:
                                try:
                                    help.input = original_in
                                except AttributeError:
                                    help._input = original_in
                            except:
                                pass
                finally:
                    sys.stdin = original_in
            except SystemExit:
                raise
            except:
                traceback.print_exc()
        finally:
            sys.__excepthook__ = sys.excepthook
        return more

    def do_add_exec(self, codeFragment):
        """
        Subclasses should override.

        @return: more (True if more input is needed to complete the statement and False if the statement is complete).
        """
        raise NotImplementedError()

    def get_namespace(self):
        """
        Subclasses should override.

        @return: dict with namespace.
        """
        raise NotImplementedError()

    def __resolve_reference__(self, text):
        """

        :type text: str
        """
        obj = None
        if '.' not in text:
            try:
                obj = self.get_namespace()[text]
            except KeyError:
                pass
            if obj is None:
                try:
                    obj = self.get_namespace()['__builtins__'][text]
                except:
                    pass
            if obj is None:
                try:
                    obj = getattr(self.get_namespace()['__builtins__'], text, None)
                except:
                    pass
        else:
            try:
                last_dot = text.rindex('.')
                parent_context = text[0:last_dot]
                res = pydevd_vars.eval_in_context(parent_context, self.get_namespace(), self.get_namespace())
                obj = getattr(res, text[last_dot + 1:])
            except:
                pass
        return obj

    def getDescription(self, text):
        try:
            obj = self.__resolve_reference__(text)
            if obj is None:
                return ''
            return get_description(obj)
        except:
            return ''

    def do_exec_code(self, code, is_single_line):
        try:
            code_fragment = CodeFragment(code, is_single_line)
            more = self.need_more(code_fragment)
            if not more:
                code_fragment = self.buffer
                self.buffer = None
                self.exec_queue.put(code_fragment)
            return more
        except:
            traceback.print_exc()
            return False

    def execLine(self, line):
        return self.do_exec_code(line, True)

    def execMultipleLines(self, lines):
        if IS_JYTHON:
            more = False
            for line in lines.split('\n'):
                more = self.do_exec_code(line, True)
            return more
        else:
            return self.do_exec_code(lines, False)

    def interrupt(self):
        self.buffer = None
        try:
            if self.interruptable:
                interrupt_main_thread(self.mainThread)
            self.finish_exec(False)
            return True
        except:
            traceback.print_exc()
            return False

    def close(self):
        sys.exit(0)

    def start_exec(self):
        self.interruptable = True

    def get_server(self):
        if getattr(self, 'host', None) is not None:
            return xmlrpclib.Server('http://%s:%s' % (self.host, self.client_port))
        else:
            return None
    server = property(get_server)

    def ShowConsole(self):
        server = self.get_server()
        if server is not None:
            server.ShowConsole()

    def finish_exec(self, more):
        self.interruptable = False
        server = self.get_server()
        if server is not None:
            return server.NotifyFinished(more)
        else:
            return True

    def getFrame(self):
        xml = StringIO()
        hidden_ns = self.get_ipython_hidden_vars_dict()
        xml.write('<xml>')
        xml.write(pydevd_xml.frame_vars_to_xml(self.get_namespace(), hidden_ns))
        xml.write('</xml>')
        return xml.getvalue()

    @silence_warnings_decorator
    def getVariable(self, attributes):
        xml = StringIO()
        xml.write('<xml>')
        val_dict = pydevd_vars.resolve_compound_var_object_fields(self.get_namespace(), attributes)
        if val_dict is None:
            val_dict = {}
        for k, val in val_dict.items():
            val = val_dict[k]
            evaluate_full_value = pydevd_xml.should_evaluate_full_value(val)
            xml.write(pydevd_vars.var_to_xml(val, k, evaluate_full_value=evaluate_full_value))
        xml.write('</xml>')
        return xml.getvalue()

    def getArray(self, attr, roffset, coffset, rows, cols, format):
        name = attr.split('\t')[-1]
        array = pydevd_vars.eval_in_context(name, self.get_namespace(), self.get_namespace())
        return pydevd_vars.table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format)

    def evaluate(self, expression):
        xml = StringIO()
        xml.write('<xml>')
        result = pydevd_vars.eval_in_context(expression, self.get_namespace(), self.get_namespace())
        xml.write(pydevd_vars.var_to_xml(result, expression))
        xml.write('</xml>')
        return xml.getvalue()

    @silence_warnings_decorator
    def loadFullValue(self, seq, scope_attrs):
        """
        Evaluate full value for async Console variables in a separate thread and send results to IDE side
        :param seq: id of command
        :param scope_attrs: a sequence of variables with their attributes separated by NEXT_VALUE_SEPARATOR
        (i.e.: obj	attr1	attr2NEXT_VALUE_SEPARATORobj2\x07ttr1	attr2)
        :return:
        """
        frame_variables = self.get_namespace()
        var_objects = []
        vars = scope_attrs.split(NEXT_VALUE_SEPARATOR)
        for var_attrs in vars:
            if '\t' in var_attrs:
                name, attrs = var_attrs.split('\t', 1)
            else:
                name = var_attrs
                attrs = None
            if name in frame_variables:
                var_object = pydevd_vars.resolve_var_object(frame_variables[name], attrs)
                var_objects.append((var_object, name))
            else:
                var_object = pydevd_vars.eval_in_context(name, frame_variables, frame_variables)
                var_objects.append((var_object, name))
        from _pydevd_bundle.pydevd_comm import GetValueAsyncThreadConsole
        py_db = getattr(self, 'debugger', None)
        if py_db is None:
            py_db = get_global_debugger()
        if py_db is None:
            from pydevd import PyDB
            py_db = PyDB()
        t = GetValueAsyncThreadConsole(py_db, self.get_server(), seq, var_objects)
        t.start()

    def changeVariable(self, attr, value):

        def do_change_variable():
            Exec('%s=%s' % (attr, value), self.get_namespace(), self.get_namespace())
        self.exec_queue.put(do_change_variable)

    def connectToDebugger(self, debuggerPort, debugger_options=None):
        """
        Used to show console with variables connection.
        Mainly, monkey-patches things in the debugger structure so that the debugger protocol works.
        """
        if debugger_options is None:
            debugger_options = {}
        env_key = 'PYDEVD_EXTRA_ENVS'
        if env_key in debugger_options:
            for env_name, value in debugger_options[env_key].items():
                existing_value = os.environ.get(env_name, None)
                if existing_value:
                    os.environ[env_name] = '%s%c%s' % (existing_value, os.path.pathsep, value)
                else:
                    os.environ[env_name] = value
                if env_name == 'PYTHONPATH':
                    sys.path.append(value)
            del debugger_options[env_key]

        def do_connect_to_debugger():
            try:
                import pydevd
                from _pydev_bundle._pydev_saved_modules import threading
            except:
                traceback.print_exc()
                sys.stderr.write('pydevd is not available, cannot connect\n')
            from _pydevd_bundle.pydevd_constants import set_thread_id
            from _pydev_bundle import pydev_localhost
            set_thread_id(threading.current_thread(), 'console_main')
            VIRTUAL_FRAME_ID = '1'
            VIRTUAL_CONSOLE_ID = 'console_main'
            f = FakeFrame()
            f.f_back = None
            f.f_globals = {}
            f.f_locals = self.get_namespace()
            self.debugger = pydevd.PyDB()
            self.debugger.add_fake_frame(thread_id=VIRTUAL_CONSOLE_ID, frame_id=VIRTUAL_FRAME_ID, frame=f)
            try:
                pydevd.apply_debugger_options(debugger_options)
                self.debugger.connect(pydev_localhost.get_localhost(), debuggerPort)
                self.debugger.prepare_to_run()
                self.debugger.disable_tracing()
            except:
                traceback.print_exc()
                sys.stderr.write('Failed to connect to target debugger.\n')
            self.debugrunning = False
            try:
                import pydevconsole
                pydevconsole.set_debug_hook(self.debugger.process_internal_commands)
            except:
                traceback.print_exc()
                sys.stderr.write('Version of Python does not support debuggable Interactive Console.\n')
        self.exec_queue.put(do_connect_to_debugger)
        return ('connect complete',)

    def handshake(self):
        if self.connect_status_queue is not None:
            self.connect_status_queue.put(True)
        return 'PyCharm'

    def get_connect_status_queue(self):
        return self.connect_status_queue

    def hello(self, input_str):
        return ('Hello eclipse',)

    def enableGui(self, guiname):
        """ Enable the GUI specified in guiname (see inputhook for list).
            As with IPython, enabling multiple GUIs isn't an error, but
            only the last one's main loop runs and it may not work
        """

        def do_enable_gui():
            from _pydev_bundle.pydev_versioncheck import versionok_for_gui
            if versionok_for_gui():
                try:
                    from pydev_ipython.inputhook import enable_gui
                    enable_gui(guiname)
                except:
                    sys.stderr.write("Failed to enable GUI event loop integration for '%s'\n" % guiname)
                    traceback.print_exc()
            elif guiname not in ['none', '', None]:
                sys.stderr.write("PyDev console: Python version does not support GUI event loop integration for '%s'\n" % guiname)
            return guiname
        self.exec_queue.put(do_enable_gui)

    def get_ipython_hidden_vars_dict(self):
        return None