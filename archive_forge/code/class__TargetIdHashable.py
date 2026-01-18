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
class _TargetIdHashable(object):

    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        if not hasattr(other, 'target'):
            return
        return other.target is self.target

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return id(self.target)