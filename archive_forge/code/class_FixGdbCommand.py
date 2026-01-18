from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class FixGdbCommand(gdb.Command):

    def __init__(self, command, actual_command):
        super(FixGdbCommand, self).__init__(command, gdb.COMMAND_DATA, gdb.COMPLETE_NONE)
        self.actual_command = actual_command

    def fix_gdb(self):
        """
        It seems that invoking either 'cy exec' and 'py-exec' work perfectly
        fine, but after this gdb's python API is entirely broken.
        Maybe some uncleared exception value is still set?
        sys.exc_clear() didn't help. A demonstration:

        (gdb) cy exec 'hello'
        'hello'
        (gdb) python gdb.execute('cont')
        RuntimeError: Cannot convert value to int.
        Error while executing Python code.
        (gdb) python gdb.execute('cont')
        [15148 refs]

        Program exited normally.
        """
        warnings.filterwarnings('ignore', '.*', RuntimeWarning, re.escape(__name__))
        try:
            int(gdb.parse_and_eval('(void *) 0')) == 0
        except RuntimeError:
            pass

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        self.fix_gdb()
        try:
            gdb.execute('%s %s' % (self.actual_command, args))
        except RuntimeError as e:
            raise gdb.GdbError(str(e))
        self.fix_gdb()