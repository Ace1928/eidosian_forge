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
def _handle_call_from_instr(self, func_name_instr, func_call_instr):
    self.load_attrs.pop(_TargetIdHashable(func_name_instr), None)
    call_name = self._getcallname(func_name_instr)
    target = None
    if not call_name:
        pass
    elif call_name in ('<listcomp>', '<genexpr>', '<setcomp>', '<dictcomp>'):
        code_obj = self.func_name_id_to_code_object[_TargetIdHashable(func_name_instr)]
        if code_obj is not None:
            children_targets = _get_smart_step_into_targets(code_obj)
            if children_targets:
                target = Target(call_name, func_name_instr.lineno, func_call_instr.offset, children_targets)
                self.function_calls.append(target)
    else:
        target = Target(call_name, func_name_instr.lineno, func_call_instr.offset)
        self.function_calls.append(target)
    if DEBUG and target is not None:
        print('Created target', target)
    self._stack.append(func_call_instr)