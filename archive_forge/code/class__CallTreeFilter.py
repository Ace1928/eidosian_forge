import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
class _CallTreeFilter:
    """Converter of a Stats object to input suitable for KCacheGrind.

    This code is taken from http://ddaa.net/blog/python/lsprof-calltree
    with the changes made by J.P. Calderone and Itamar applied. Note that
    isinstance(code, str) needs to be used at times to determine if the code
    object is actually an external code object (with a filename, etc.) or
    a Python built-in.
    """
    out_file: TextIO

    def __init__(self, data):
        self.data = data

    def output(self, out_file):
        self.out_file = out_file
        out_file.write('events: Ticks\n')
        self._print_summary()
        for entry in self.data:
            self._entry(entry)

    def _print_summary(self):
        max_cost = 0
        for entry in self.data:
            totaltime = int(entry.totaltime * 1000)
            max_cost = max(max_cost, totaltime)
        self.out_file.write('summary: %d\n' % (max_cost,))

    def _entry(self, entry):
        out_file = self.out_file
        code = entry.code
        inlinetime = int(entry.inlinetime * 1000)
        if isinstance(code, str):
            out_file.write('fi=~\n')
        else:
            out_file.write('fi={}\n'.format(code.co_filename))
        out_file.write('fn={}\n'.format(label(code, True)))
        if isinstance(code, str):
            out_file.write('0  {}\n'.format(inlinetime))
        else:
            out_file.write('%d %d\n' % (code.co_firstlineno, inlinetime))
        if entry.calls:
            calls = entry.calls
        else:
            calls = []
        if isinstance(code, str):
            lineno = 0
        else:
            lineno = code.co_firstlineno
        for subentry in calls:
            self._subentry(lineno, subentry)
        out_file.write('\n')

    def _subentry(self, lineno, subentry):
        out_file = self.out_file
        code = subentry.code
        totaltime = int(subentry.totaltime * 1000)
        if isinstance(code, str):
            out_file.write('cfi=~\n')
            out_file.write('cfn={}\n'.format(label(code, True)))
            out_file.write('calls=%d 0\n' % (subentry.callcount,))
        else:
            out_file.write('cfi={}\n'.format(code.co_filename))
            out_file.write('cfn={}\n'.format(label(code, True)))
            out_file.write('calls=%d %d\n' % (subentry.callcount, code.co_firstlineno))
        out_file.write('%d %d\n' % (lineno, totaltime))