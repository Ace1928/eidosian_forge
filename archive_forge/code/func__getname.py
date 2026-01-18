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
def _getname(self, instr):
    if instr.opcode in _opcode.hascompare:
        cmp_op = dis.cmp_op[instr.arg]
        if cmp_op not in ('exception match', 'BAD'):
            return _COMP_OP_MAP.get(cmp_op, cmp_op)
    return instr.arg