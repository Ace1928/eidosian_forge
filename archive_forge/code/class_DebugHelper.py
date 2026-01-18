from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break
class DebugHelper(object):

    def __init__(self):
        self._debug_dir = os.path.join(os.path.dirname(__file__), 'debug_info')
        try:
            os.makedirs(self._debug_dir)
        except:
            pass
        self._next = partial(next, itertools.count(0))

    def _get_filename(self, op_number=None, prefix=''):
        if op_number is None:
            op_number = self._next()
            name = '%03d_before.txt' % op_number
        else:
            name = '%03d_change.txt' % op_number
        filename = os.path.join(self._debug_dir, prefix + name)
        return (filename, op_number)

    def write_bytecode(self, b, op_number=None, prefix=''):
        filename, op_number = self._get_filename(op_number, prefix)
        with open(filename, 'w') as stream:
            bytecode.dump_bytecode(b, stream=stream, lineno=True)
        return op_number

    def write_dis(self, code_to_modify, op_number=None, prefix=''):
        filename, op_number = self._get_filename(op_number, prefix)
        with open(filename, 'w') as stream:
            stream.write('-------- ')
            stream.write('-------- ')
            stream.write('id(code_to_modify): %s' % id(code_to_modify))
            stream.write('\n\n')
            dis.dis(code_to_modify, file=stream)
        return op_number