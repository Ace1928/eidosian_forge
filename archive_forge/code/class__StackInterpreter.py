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
class _StackInterpreter(object):
    """
    Good reference: https://github.com/python/cpython/blob/fcb55c0037baab6f98f91ee38ce84b6f874f034a/Python/ceval.c
    """

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self._stack = deque()
        self.function_calls = []
        self.load_attrs = {}
        self.func = set()
        self.func_name_id_to_code_object = {}

    def __str__(self):
        return 'Stack:\nFunction calls:\n%s\nLoad attrs:\n%s\n' % (self.function_calls, list(self.load_attrs.values()))

    def _getname(self, instr):
        if instr.opcode in _opcode.hascompare:
            cmp_op = dis.cmp_op[instr.arg]
            if cmp_op not in ('exception match', 'BAD'):
                return _COMP_OP_MAP.get(cmp_op, cmp_op)
        return instr.arg

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

    def _no_stack_change(self, instr):
        pass

    def on_LOAD_GLOBAL(self, instr):
        self._stack.append(instr)

    def on_POP_TOP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            pass

    def on_LOAD_ATTR(self, instr):
        self.on_POP_TOP(instr)
        self._stack.append(instr)
        self.load_attrs[_TargetIdHashable(instr)] = Target(self._getname(instr), instr.lineno, instr.offset)
    on_LOOKUP_METHOD = on_LOAD_ATTR

    def on_LOAD_CONST(self, instr):
        self._stack.append(instr)
    on_LOAD_DEREF = on_LOAD_CONST
    on_LOAD_NAME = on_LOAD_CONST
    on_LOAD_CLOSURE = on_LOAD_CONST
    on_LOAD_CLASSDEREF = on_LOAD_CONST
    on_IMPORT_NAME = _no_stack_change
    on_IMPORT_FROM = _no_stack_change
    on_IMPORT_STAR = _no_stack_change
    on_SETUP_ANNOTATIONS = _no_stack_change

    def on_STORE_FAST(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            pass
    on_STORE_GLOBAL = on_STORE_FAST
    on_STORE_DEREF = on_STORE_FAST
    on_STORE_ATTR = on_STORE_FAST
    on_STORE_NAME = on_STORE_FAST
    on_DELETE_NAME = on_POP_TOP
    on_DELETE_ATTR = on_POP_TOP
    on_DELETE_GLOBAL = on_POP_TOP
    on_DELETE_FAST = on_POP_TOP
    on_DELETE_DEREF = on_POP_TOP
    on_DICT_UPDATE = on_POP_TOP
    on_SET_UPDATE = on_POP_TOP
    on_GEN_START = on_POP_TOP

    def on_NOP(self, instr):
        pass

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

    def on_COMPARE_OP(self, instr):
        try:
            _right = self._stack.pop()
        except IndexError:
            return
        try:
            _left = self._stack.pop()
        except IndexError:
            return
        cmp_op = dis.cmp_op[instr.arg]
        if cmp_op not in ('exception match', 'BAD'):
            self.function_calls.append(Target(self._getname(instr), instr.lineno, instr.offset))
        self._stack.append(instr)

    def on_IS_OP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return
        try:
            self._stack.pop()
        except IndexError:
            return

    def on_BINARY_SUBSCR(self, instr):
        try:
            _sub = self._stack.pop()
        except IndexError:
            return
        try:
            _container = self._stack.pop()
        except IndexError:
            return
        self.function_calls.append(Target(_BINARY_OP_MAP[instr.name], instr.lineno, instr.offset))
        self._stack.append(instr)
    on_BINARY_MATRIX_MULTIPLY = on_BINARY_SUBSCR
    on_BINARY_POWER = on_BINARY_SUBSCR
    on_BINARY_MULTIPLY = on_BINARY_SUBSCR
    on_BINARY_FLOOR_DIVIDE = on_BINARY_SUBSCR
    on_BINARY_TRUE_DIVIDE = on_BINARY_SUBSCR
    on_BINARY_MODULO = on_BINARY_SUBSCR
    on_BINARY_ADD = on_BINARY_SUBSCR
    on_BINARY_SUBTRACT = on_BINARY_SUBSCR
    on_BINARY_LSHIFT = on_BINARY_SUBSCR
    on_BINARY_RSHIFT = on_BINARY_SUBSCR
    on_BINARY_AND = on_BINARY_SUBSCR
    on_BINARY_OR = on_BINARY_SUBSCR
    on_BINARY_XOR = on_BINARY_SUBSCR

    def on_LOAD_METHOD(self, instr):
        self.on_POP_TOP(instr)
        self._stack.append(instr)

    def on_MAKE_FUNCTION(self, instr):
        if not IS_PY311_OR_GREATER:
            qualname = self._stack.pop()
            code_obj_instr = self._stack.pop()
        else:
            qualname = code_obj_instr = self._stack.pop()
        arg = instr.arg
        if arg & 8:
            _func_closure = self._stack.pop()
        if arg & 4:
            _func_annotations = self._stack.pop()
        if arg & 2:
            _func_kwdefaults = self._stack.pop()
        if arg & 1:
            _func_defaults = self._stack.pop()
        call_name = self._getcallname(qualname)
        if call_name in ('<genexpr>', '<listcomp>', '<setcomp>', '<dictcomp>'):
            if isinstance(code_obj_instr.arg, CodeType):
                self.func_name_id_to_code_object[_TargetIdHashable(qualname)] = code_obj_instr.arg
        self._stack.append(qualname)

    def on_LOAD_FAST(self, instr):
        self._stack.append(instr)

    def on_LOAD_ASSERTION_ERROR(self, instr):
        self._stack.append(instr)
    on_LOAD_BUILD_CLASS = on_LOAD_FAST

    def on_CALL_METHOD(self, instr):
        for _ in range(instr.arg):
            self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_PUSH_NULL(self, instr):
        self._stack.append(instr)

    def on_CALL_FUNCTION(self, instr):
        arg = instr.arg
        argc = arg & 255
        argc += (arg >> 8) * 2
        for _ in range(argc):
            try:
                self._stack.pop()
            except IndexError:
                return
        try:
            func_name_instr = self._stack.pop()
        except IndexError:
            return
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_KW(self, instr):
        _names_of_kw_args = self._stack.pop()
        arg = instr.arg
        argc = arg & 255
        argc += (arg >> 8) * 2
        for _ in range(argc):
            self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_VAR(self, instr):
        _var_arg = self._stack.pop()
        arg = instr.arg
        argc = arg & 255
        argc += (arg >> 8) * 2
        for _ in range(argc):
            self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_VAR_KW(self, instr):
        _names_of_kw_args = self._stack.pop()
        arg = instr.arg
        argc = arg & 255
        argc += (arg >> 8) * 2
        self._stack.pop()
        for _ in range(argc):
            self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)

    def on_CALL_FUNCTION_EX(self, instr):
        if instr.arg & 1:
            _kwargs = self._stack.pop()
        _callargs = self._stack.pop()
        func_name_instr = self._stack.pop()
        self._handle_call_from_instr(func_name_instr, instr)
    on_YIELD_VALUE = _no_stack_change
    on_GET_AITER = _no_stack_change
    on_GET_ANEXT = _no_stack_change
    on_END_ASYNC_FOR = _no_stack_change
    on_BEFORE_ASYNC_WITH = _no_stack_change
    on_SETUP_ASYNC_WITH = _no_stack_change
    on_YIELD_FROM = _no_stack_change
    on_SETUP_LOOP = _no_stack_change
    on_FOR_ITER = _no_stack_change
    on_BREAK_LOOP = _no_stack_change
    on_JUMP_ABSOLUTE = _no_stack_change
    on_RERAISE = _no_stack_change
    on_LIST_TO_TUPLE = _no_stack_change
    on_CALL_FINALLY = _no_stack_change
    on_POP_FINALLY = _no_stack_change

    def on_JUMP_IF_FALSE_OR_POP(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return
    on_JUMP_IF_TRUE_OR_POP = on_JUMP_IF_FALSE_OR_POP

    def on_JUMP_IF_NOT_EXC_MATCH(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            return
        try:
            self._stack.pop()
        except IndexError:
            return

    def on_ROT_TWO(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return
        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return
        self._stack.append(p0)
        self._stack.append(p1)

    def on_ROT_THREE(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return
        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return
        try:
            p2 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            return
        self._stack.append(p0)
        self._stack.append(p1)
        self._stack.append(p2)

    def on_ROT_FOUR(self, instr):
        try:
            p0 = self._stack.pop()
        except IndexError:
            return
        try:
            p1 = self._stack.pop()
        except:
            self._stack.append(p0)
            return
        try:
            p2 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            return
        try:
            p3 = self._stack.pop()
        except:
            self._stack.append(p0)
            self._stack.append(p1)
            self._stack.append(p2)
            return
        self._stack.append(p0)
        self._stack.append(p1)
        self._stack.append(p2)
        self._stack.append(p3)

    def on_BUILD_LIST_FROM_ARG(self, instr):
        self._stack.append(instr)

    def on_BUILD_MAP(self, instr):
        for _i in range(instr.arg):
            self._stack.pop()
            self._stack.pop()
        self._stack.append(instr)

    def on_BUILD_CONST_KEY_MAP(self, instr):
        self.on_POP_TOP(instr)
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)
        self._stack.append(instr)
    on_RETURN_VALUE = on_POP_TOP
    on_POP_JUMP_IF_FALSE = on_POP_TOP
    on_POP_JUMP_IF_TRUE = on_POP_TOP
    on_DICT_MERGE = on_POP_TOP
    on_LIST_APPEND = on_POP_TOP
    on_SET_ADD = on_POP_TOP
    on_LIST_EXTEND = on_POP_TOP
    on_UNPACK_EX = on_POP_TOP
    on_GET_ITER = _no_stack_change
    on_GET_AWAITABLE = _no_stack_change
    on_GET_YIELD_FROM_ITER = _no_stack_change

    def on_RETURN_GENERATOR(self, instr):
        self._stack.append(instr)
    on_RETURN_GENERATOR = _no_stack_change
    on_RESUME = _no_stack_change

    def on_MAP_ADD(self, instr):
        self.on_POP_TOP(instr)
        self.on_POP_TOP(instr)

    def on_UNPACK_SEQUENCE(self, instr):
        self._stack.pop()
        for _i in range(instr.arg):
            self._stack.append(instr)

    def on_BUILD_LIST(self, instr):
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)
        self._stack.append(instr)
    on_BUILD_TUPLE = on_BUILD_LIST
    on_BUILD_STRING = on_BUILD_LIST
    on_BUILD_TUPLE_UNPACK_WITH_CALL = on_BUILD_LIST
    on_BUILD_TUPLE_UNPACK = on_BUILD_LIST
    on_BUILD_LIST_UNPACK = on_BUILD_LIST
    on_BUILD_MAP_UNPACK_WITH_CALL = on_BUILD_LIST
    on_BUILD_MAP_UNPACK = on_BUILD_LIST
    on_BUILD_SET = on_BUILD_LIST
    on_BUILD_SET_UNPACK = on_BUILD_LIST
    on_SETUP_FINALLY = _no_stack_change
    on_POP_FINALLY = _no_stack_change
    on_BEGIN_FINALLY = _no_stack_change
    on_END_FINALLY = _no_stack_change

    def on_RAISE_VARARGS(self, instr):
        for _i in range(instr.arg):
            self.on_POP_TOP(instr)
    on_POP_BLOCK = _no_stack_change
    on_JUMP_FORWARD = _no_stack_change
    on_POP_EXCEPT = _no_stack_change
    on_SETUP_EXCEPT = _no_stack_change
    on_WITH_EXCEPT_START = _no_stack_change
    on_END_FINALLY = _no_stack_change
    on_BEGIN_FINALLY = _no_stack_change
    on_SETUP_WITH = _no_stack_change
    on_WITH_CLEANUP_START = _no_stack_change
    on_WITH_CLEANUP_FINISH = _no_stack_change
    on_FORMAT_VALUE = _no_stack_change
    on_EXTENDED_ARG = _no_stack_change

    def on_INPLACE_ADD(self, instr):
        try:
            self._stack.pop()
        except IndexError:
            pass
    on_INPLACE_POWER = on_INPLACE_ADD
    on_INPLACE_MULTIPLY = on_INPLACE_ADD
    on_INPLACE_MATRIX_MULTIPLY = on_INPLACE_ADD
    on_INPLACE_TRUE_DIVIDE = on_INPLACE_ADD
    on_INPLACE_FLOOR_DIVIDE = on_INPLACE_ADD
    on_INPLACE_MODULO = on_INPLACE_ADD
    on_INPLACE_SUBTRACT = on_INPLACE_ADD
    on_INPLACE_RSHIFT = on_INPLACE_ADD
    on_INPLACE_LSHIFT = on_INPLACE_ADD
    on_INPLACE_AND = on_INPLACE_ADD
    on_INPLACE_OR = on_INPLACE_ADD
    on_INPLACE_XOR = on_INPLACE_ADD

    def on_DUP_TOP(self, instr):
        try:
            i = self._stack[-1]
        except IndexError:
            self._stack.append(instr)
        else:
            self._stack.append(i)

    def on_DUP_TOP_TWO(self, instr):
        if len(self._stack) == 0:
            self._stack.append(instr)
            return
        if len(self._stack) == 1:
            i = self._stack[-1]
            self._stack.append(i)
            self._stack.append(instr)
            return
        i = self._stack[-1]
        j = self._stack[-2]
        self._stack.append(j)
        self._stack.append(i)

    def on_BUILD_SLICE(self, instr):
        for _ in range(instr.arg):
            try:
                self._stack.pop()
            except IndexError:
                pass
        self._stack.append(instr)

    def on_STORE_SUBSCR(self, instr):
        try:
            self._stack.pop()
            self._stack.pop()
            self._stack.pop()
        except IndexError:
            pass

    def on_DELETE_SUBSCR(self, instr):
        try:
            self._stack.pop()
            self._stack.pop()
        except IndexError:
            pass
    on_PRINT_EXPR = on_POP_TOP
    on_UNARY_POSITIVE = _no_stack_change
    on_UNARY_NEGATIVE = _no_stack_change
    on_UNARY_NOT = _no_stack_change
    on_UNARY_INVERT = _no_stack_change
    on_CACHE = _no_stack_change
    on_PRECALL = _no_stack_change