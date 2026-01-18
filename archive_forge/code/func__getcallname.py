from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode
from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque
def _getcallname(self, instr):
    if instr.name == 'BINARY_SUBSCR':
        return '__getitem__().__call__'
    if instr.name == 'CALL_FUNCTION':
        return None
    if instr.name == 'MAKE_FUNCTION':
        return '__func__().__call__'
    if instr.name == 'LOAD_ASSERTION_ERROR':
        return 'AssertionError'
    name = self._getname(instr)
    if isinstance(name, CodeType):
        name = name.co_qualname
    if isinstance(name, _Variable):
        name = name.name
    if not isinstance(name, str):
        return None
    if name.endswith('>'):
        return name.split('.')[-1]
    return name