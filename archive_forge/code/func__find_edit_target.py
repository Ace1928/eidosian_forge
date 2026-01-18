import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list
@staticmethod
def _find_edit_target(shell, args, opts, last_call):
    """Utility method used by magic_edit to find what to edit."""

    def make_filename(arg):
        """Make a filename from the given args"""
        try:
            filename = get_py_filename(arg)
        except IOError:
            if arg.endswith('.py'):
                filename = arg
            else:
                filename = None
        return filename
    opts_prev = 'p' in opts
    opts_raw = 'r' in opts

    class DataIsObject(Exception):
        pass
    lineno = opts.get('n', None)
    if opts_prev:
        args = '_%s' % last_call[0]
        if args not in shell.user_ns:
            args = last_call[1]
    use_temp = True
    data = ''
    filename = make_filename(args)
    if filename:
        use_temp = False
    elif args:
        data = shell.extract_input_lines(args, opts_raw)
        if not data:
            try:
                data = eval(args, shell.user_ns)
                if not isinstance(data, str):
                    raise DataIsObject
            except (NameError, SyntaxError):
                filename = make_filename(args)
                if filename is None:
                    warn("Argument given (%s) can't be found as a variable or as a filename." % args)
                    return (None, None, None)
                use_temp = False
            except DataIsObject as e:
                if isinstance(data, Macro):
                    raise MacroToEdit(data) from e
                filename = find_file(data)
                if filename:
                    if 'fakemodule' in filename.lower() and inspect.isclass(data):
                        attrs = [getattr(data, aname) for aname in dir(data)]
                        for attr in attrs:
                            if not inspect.ismethod(attr):
                                continue
                            filename = find_file(attr)
                            if filename and 'fakemodule' not in filename.lower():
                                data = attr
                                break
                    m = ipython_input_pat.match(os.path.basename(filename))
                    if m:
                        raise InteractivelyDefined(int(m.groups()[0])) from e
                    datafile = 1
                if filename is None:
                    filename = make_filename(args)
                    datafile = 1
                    if filename is not None:
                        warn('Could not find file where `%s` is defined.\nOpening a file named `%s`' % (args, filename))
                if datafile:
                    if lineno is None:
                        lineno = find_source_lines(data)
                    if lineno is None:
                        filename = make_filename(args)
                        if filename is None:
                            warn('The file where `%s` was defined cannot be read or found.' % data)
                            return (None, None, None)
                use_temp = False
    if use_temp:
        filename = shell.mktempfile(data)
        print('IPython will make a temporary file named:', filename)
    try:
        last_call[0] = shell.displayhook.prompt_count
        if not opts_prev:
            last_call[1] = args
    except:
        pass
    return (filename, lineno, use_temp)