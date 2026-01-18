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
def _get_smart_step_into_targets(code):
    """
    :return list(Target)
    """
    b = bytecode.Bytecode.from_code(code)
    cfg = bytecode_cfg.ControlFlowGraph.from_bytecode(b)
    ret = []
    for block in cfg:
        if DEBUG:
            print('\nStart block----')
        stack = _StackInterpreter(block)
        for instr in block:
            try:
                func_name = 'on_%s' % (instr.name,)
                func = getattr(stack, func_name, None)
                if DEBUG:
                    if instr.name != 'CACHE':
                        print('\nWill handle: ', instr, '>>', stack._getname(instr), '<<')
                        print('Current stack:')
                        for entry in stack._stack:
                            print('    arg:', stack._getname(entry), '(', entry, ')')
                if func is None:
                    if STRICT_MODE:
                        raise AssertionError('%s not found.' % (func_name,))
                    else:
                        continue
                func(instr)
            except:
                if STRICT_MODE:
                    raise
                elif DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
                    pydev_log.exception('Exception computing step into targets (handled).')
        ret.extend(stack.function_calls)
    return ret