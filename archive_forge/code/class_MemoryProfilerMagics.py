from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
@magics_class
class MemoryProfilerMagics(Magics):

    @line_cell_magic
    def mprun(self, parameter_s='', cell=None):
        """ Execute a statement under the line-by-line memory profiler from the
        memory_profiler module.

        Usage, in line mode:
          %mprun -f func1 -f func2 <statement>

        Usage, in cell mode:
          %%mprun -f func1 -f func2 [statement]
          code...
          code...

        In cell mode, the additional code lines are appended to the (possibly
        empty) statement in the first line. Cell mode allows you to easily
        profile multiline blocks without having to put them in a separate
        function.

        The given statement (which doesn't require quote marks) is run via the
        LineProfiler. Profiling is enabled for the functions specified by the -f
        options. The statistics will be shown side-by-side with the code through
        the pager once the statement has completed.

        Options:

        -f <function>: LineProfiler only profiles functions and methods it is told
        to profile.  This option tells the profiler about these functions. Multiple
        -f options may be used. The argument may be any expression that gives
        a Python function or method object. However, one must be careful to avoid
        spaces that may confuse the option parser. Additionally, functions defined
        in the interpreter at the In[] prompt or via %run currently cannot be
        displayed.  Write these functions out to a separate file and import them.

        One or more -f options are required to get any useful results.

        -T <filename>: dump the text-formatted statistics with the code
        side-by-side out to a text file.

        -r: return the LineProfiler object after it has completed profiling.

        -c: If present, add the memory usage of any children process to the report.
        """
        from io import StringIO
        from memory_profiler import show_results, LineProfiler
        from distutils.version import LooseVersion
        import IPython
        ipython_version = LooseVersion(IPython.__version__)
        if ipython_version < '0.11':
            from IPython.genutils import page
            from IPython.ipstruct import Struct
            from IPython.ipapi import UsageError
        else:
            from IPython.core.page import page
            from IPython.utils.ipstruct import Struct
            from IPython.core.error import UsageError
        opts_def = Struct(T=[''], f=[])
        parameter_s = parameter_s.replace('"', '\\"').replace("'", "\\'")
        opts, arg_str = self.parse_options(parameter_s, 'rf:T:c', list_all=True)
        opts.merge(opts_def)
        global_ns = self.shell.user_global_ns
        local_ns = self.shell.user_ns
        if cell is not None:
            arg_str += '\n' + cell
        funcs = []
        for name in opts.f:
            try:
                funcs.append(eval(name, global_ns, local_ns))
            except Exception as e:
                raise UsageError('Could not find function %r.\n%s: %s' % (name, e.__class__.__name__, e))
        include_children = 'c' in opts
        profile = LineProfiler(include_children=include_children)
        for func in funcs:
            profile(func)
        if 'profile' in builtins.__dict__:
            had_profile = True
            old_profile = builtins.__dict__['profile']
        else:
            had_profile = False
            old_profile = None
        builtins.__dict__['profile'] = profile
        try:
            profile.runctx(arg_str, global_ns, local_ns)
            message = ''
        except SystemExit:
            message = '*** SystemExit exception caught in code being profiled.'
        except KeyboardInterrupt:
            message = '*** KeyboardInterrupt exception caught in code being profiled.'
        finally:
            if had_profile:
                builtins.__dict__['profile'] = old_profile
        stdout_trap = StringIO()
        show_results(profile, stdout_trap)
        output = stdout_trap.getvalue()
        output = output.rstrip()
        if ipython_version < '0.11':
            page(output, screen_lines=self.shell.rc.screen_length)
        else:
            page(output)
        print(message)
        text_file = opts.T[0]
        if text_file:
            with open(text_file, 'w') as pfile:
                pfile.write(output)
            print('\n*** Profile printout saved to text file %s. %s' % (text_file, message))
        return_value = None
        if 'r' in opts:
            return_value = profile
        return return_value

    @line_cell_magic
    def memit(self, line='', cell=None):
        """Measure memory usage of a Python statement

        Usage, in line mode:
          %memit [-r<R>t<T>i<I>] statement

        Usage, in cell mode:
          %%memit [-r<R>t<T>i<I>] setup_code
          code...
          code...

        This function can be used both as a line and cell magic:

        - In line mode you can measure a single-line statement (though multiple
          ones can be chained with using semicolons).

        - In cell mode, the statement in the first line is used as setup code
          (executed but not measured) and the body of the cell is measured.
          The cell body has access to any variables created in the setup code.

        Options:
        -r<R>: repeat the loop iteration <R> times and take the best result.
        Default: 1

        -t<T>: timeout after <T> seconds. Default: None

        -i<I>: Get time information at an interval of I times per second.
            Defaults to 0.1 so that there is ten measurements per second.

        -c: If present, add the memory usage of any children process to the report.

        -o: If present, return a object containing memit run details

        -q: If present, be quiet and do not output a result.

        Examples
        --------
        ::

          In [1]: %memit range(10000)
          peak memory: 21.42 MiB, increment: 0.41 MiB

          In [2]: %memit range(1000000)
          peak memory: 52.10 MiB, increment: 31.08 MiB

          In [3]: %%memit l=range(1000000)
             ...: len(l)
             ...:
          peak memory: 52.14 MiB, increment: 0.08 MiB

        """
        from memory_profiler import memory_usage, _func_exec
        opts, stmt = self.parse_options(line, 'r:t:i:coq', posix=False, strict=False)
        if cell is None:
            setup = 'pass'
        else:
            setup = stmt
            stmt = cell
        repeat = int(getattr(opts, 'r', 1))
        if repeat < 1:
            repeat == 1
        timeout = int(getattr(opts, 't', 0))
        if timeout <= 0:
            timeout = None
        interval = float(getattr(opts, 'i', 0.1))
        include_children = 'c' in opts
        return_result = 'o' in opts
        quiet = 'q' in opts
        import gc
        gc.collect()
        _func_exec(setup, self.shell.user_ns)
        mem_usage = []
        counter = 0
        baseline = memory_usage()[0]
        while counter < repeat:
            counter += 1
            tmp = memory_usage((_func_exec, (stmt, self.shell.user_ns)), timeout=timeout, interval=interval, max_usage=True, max_iterations=1, include_children=include_children)
            mem_usage.append(tmp)
        result = MemitResult(mem_usage, baseline, repeat, timeout, interval, include_children)
        if not quiet:
            if mem_usage:
                print(result)
            else:
                print('ERROR: could not read memory usage, try with a lower interval or more iterations')
        if return_result:
            return result

    @classmethod
    def register_magics(cls, ip):
        from distutils.version import LooseVersion
        import IPython
        ipython_version = LooseVersion(IPython.__version__)
        if ipython_version < '0.13':
            try:
                _register_magic = ip.define_magic
            except AttributeError:
                _register_magic = ip.expose_magic
            _register_magic('mprun', cls.mprun.__func__)
            _register_magic('memit', cls.memit.__func__)
        else:
            ip.register_magics(cls)