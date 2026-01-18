import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _GetIter(_BaseHandler):
    """
    Implements TOS = iter(TOS).
    """
    opname = 'GET_ITER'
    iter_target = None

    def _handle(self):
        self.iter_target = self.stack.pop()
        self.tokens.extend(self.iter_target.tokens)
        self.stack.push(self)