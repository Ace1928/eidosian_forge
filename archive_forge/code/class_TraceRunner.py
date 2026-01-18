import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class TraceRunner(object):
    """Trace runner contains the states for the trace and the opcode dispatch.
    """

    def __init__(self, debug_filename):
        self.debug_filename = debug_filename
        self.pending = deque()
        self.finished = set()

    def get_debug_loc(self, lineno):
        return Loc(self.debug_filename, lineno)

    def dispatch(self, state):
        if PYVERSION in ((3, 11), (3, 12)):
            if state._blockstack:
                state: State
                while state._blockstack:
                    topblk = state._blockstack[-1]
                    blk_end = topblk['end']
                    if blk_end is not None and blk_end <= state.pc_initial:
                        state._blockstack.pop()
                    else:
                        break
        elif PYVERSION in ((3, 9), (3, 10)):
            pass
        else:
            raise NotImplementedError(PYVERSION)
        inst = state.get_inst()
        if inst.opname != 'CACHE':
            _logger.debug('dispatch pc=%s, inst=%s', state._pc, inst)
            _logger.debug('stack %s', state._stack)
        fn = getattr(self, 'op_{}'.format(inst.opname), None)
        if fn is not None:
            fn(state, inst)
        else:
            msg = 'Use of unsupported opcode (%s) found' % inst.opname
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def _adjust_except_stack(self, state):
        """
        Adjust stack when entering an exception handler to match expectation
        by the bytecode.
        """
        tryblk = state.get_top_block('TRY')
        state.pop_block_and_above(tryblk)
        nstack = state.stack_depth
        kwargs = {}
        expected_depth = tryblk['stack_depth']
        if nstack > expected_depth:
            kwargs['npop'] = nstack - expected_depth
        extra_stack = 1
        if tryblk['push_lasti']:
            extra_stack += 1
        kwargs['npush'] = extra_stack
        state.fork(pc=tryblk['end'], **kwargs)

    def op_NOP(self, state, inst):
        state.append(inst)

    def op_RESUME(self, state, inst):
        state.append(inst)

    def op_CACHE(self, state, inst):
        state.append(inst)

    def op_PRECALL(self, state, inst):
        state.append(inst)

    def op_PUSH_NULL(self, state, inst):
        state.push(state.make_null())
        state.append(inst)

    def op_RETURN_GENERATOR(self, state, inst):
        state.push(state.make_temp())
        state.append(inst)

    def op_FORMAT_VALUE(self, state, inst):
        """
        FORMAT_VALUE(flags): flags argument specifies format spec which is
        not supported yet. Currently, we just call str() on the value.
        Pops a value from stack and pushes results back.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE
        """
        if inst.arg != 0:
            msg = 'format spec in f-strings not supported yet'
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))
        value = state.pop()
        strvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, value=value, res=res, strvar=strvar)
        state.push(res)

    def op_BUILD_STRING(self, state, inst):
        """
        BUILD_STRING(count): Concatenates count strings from the stack and
        pushes the resulting string onto the stack.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
        """
        count = inst.arg
        strings = list(reversed([state.pop() for _ in range(count)]))
        if count == 0:
            tmps = [state.make_temp()]
        else:
            tmps = [state.make_temp() for _ in range(count - 1)]
        state.append(inst, strings=strings, tmps=tmps)
        state.push(tmps[-1])

    def op_POP_TOP(self, state, inst):
        state.pop()
    if PYVERSION in ((3, 11), (3, 12)):

        def op_LOAD_GLOBAL(self, state, inst):
            res = state.make_temp()
            idx = inst.arg >> 1
            state.append(inst, idx=idx, res=res)
            if inst.arg & 1:
                state.push(state.make_null())
            state.push(res)
    elif PYVERSION in ((3, 9), (3, 10)):

        def op_LOAD_GLOBAL(self, state, inst):
            res = state.make_temp()
            state.append(inst, res=res)
            state.push(res)
    else:
        raise NotImplementedError(PYVERSION)

    def op_COPY_FREE_VARS(self, state, inst):
        state.append(inst)

    def op_MAKE_CELL(self, state, inst):
        state.append(inst)

    def op_LOAD_DEREF(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_CONST(self, state, inst):
        res = state.make_temp('const')
        state.push(res)
        state.append(inst, res=res)

    def op_LOAD_ATTR(self, state, inst):
        item = state.pop()
        if PYVERSION in ((3, 12),):
            if inst.arg & 1:
                state.push(state.make_null())
        elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
            pass
        else:
            raise NotImplementedError(PYVERSION)
        res = state.make_temp()
        state.append(inst, item=item, res=res)
        state.push(res)

    def op_LOAD_FAST(self, state, inst):
        name = state.get_varname(inst)
        res = state.make_temp(name)
        state.append(inst, res=res)
        state.push(res)
    if PYVERSION in ((3, 12),):
        op_LOAD_FAST_CHECK = op_LOAD_FAST
        op_LOAD_FAST_AND_CLEAR = op_LOAD_FAST
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def op_DELETE_FAST(self, state, inst):
        state.append(inst)

    def op_DELETE_ATTR(self, state, inst):
        target = state.pop()
        state.append(inst, target=target)

    def op_STORE_ATTR(self, state, inst):
        target = state.pop()
        value = state.pop()
        state.append(inst, target=target, value=value)

    def op_STORE_DEREF(self, state, inst):
        value = state.pop()
        state.append(inst, value=value)

    def op_STORE_FAST(self, state, inst):
        value = state.pop()
        state.append(inst, value=value)

    def op_SLICE_1(self, state, inst):
        """
        TOS = TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, res=res, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_2(self, state, inst):
        """
        TOS = TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_3(self, state, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, res=res, slicevar=slicevar, indexvar=indexvar)
        state.push(res)

    def op_STORE_SLICE_0(self, state, inst):
        """
        TOS[:] = TOS1
        """
        tos = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, value=value, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_1(self, state, inst):
        """
        TOS1[TOS:] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar, value=value, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_2(self, state, inst):
        """
        TOS1[:TOS] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, value=value, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_3(self, state, inst):
        """
        TOS2[TOS1:TOS] = TOS3
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, value=value, slicevar=slicevar, indexvar=indexvar)

    def op_DELETE_SLICE_0(self, state, inst):
        """
        del TOS[:]
        """
        tos = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_1(self, state, inst):
        """
        del TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_2(self, state, inst):
        """
        del TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_3(self, state, inst):
        """
        del TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, slicevar=slicevar, indexvar=indexvar)

    def op_BUILD_SLICE(self, state, inst):
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
        argc = inst.arg
        if argc == 2:
            tos = state.pop()
            tos1 = state.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = state.pop()
            tos1 = state.pop()
            tos2 = state.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception('unreachable')
        slicevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, start=start, stop=stop, step=step, res=res, slicevar=slicevar)
        state.push(res)
    if PYVERSION in ((3, 12),):

        def op_BINARY_SLICE(self, state, inst):
            end = state.pop()
            start = state.pop()
            container = state.pop()
            temp_res = state.make_temp()
            res = state.make_temp()
            slicevar = state.make_temp()
            state.append(inst, start=start, end=end, container=container, res=res, slicevar=slicevar, temp_res=temp_res)
            state.push(res)
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)
    if PYVERSION in ((3, 12),):

        def op_STORE_SLICE(self, state, inst):
            end = state.pop()
            start = state.pop()
            container = state.pop()
            value = state.pop()
            slicevar = state.make_temp()
            res = state.make_temp()
            state.append(inst, start=start, end=end, container=container, value=value, res=res, slicevar=slicevar)
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def _op_POP_JUMP_IF(self, state, inst):
        pred = state.pop()
        state.append(inst, pred=pred)
        target_inst = inst.get_jump_target()
        next_inst = inst.next
        state.fork(pc=next_inst)
        if target_inst != next_inst:
            state.fork(pc=target_inst)
    op_POP_JUMP_IF_TRUE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_FALSE = _op_POP_JUMP_IF
    if PYVERSION in ((3, 12),):
        op_POP_JUMP_IF_NONE = _op_POP_JUMP_IF
        op_POP_JUMP_IF_NOT_NONE = _op_POP_JUMP_IF
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def _op_JUMP_IF_OR_POP(self, state, inst):
        pred = state.get_tos()
        state.append(inst, pred=pred)
        state.fork(pc=inst.next, npop=1)
        state.fork(pc=inst.get_jump_target())
    op_JUMP_IF_FALSE_OR_POP = _op_JUMP_IF_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_JUMP_IF_OR_POP

    def op_POP_JUMP_FORWARD_IF_NONE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_NONE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_NOT_NONE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_FALSE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_TRUE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, state, inst):
        self._op_POP_JUMP_IF(state, inst)

    def op_JUMP_FORWARD(self, state, inst):
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_JUMP_BACKWARD(self, state, inst):
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_JUMP_ABSOLUTE(self, state, inst):
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_BREAK_LOOP(self, state, inst):
        end = state.get_top_block('LOOP')['end']
        state.append(inst, end=end)
        state.pop_block()
        state.fork(pc=end)

    def op_RETURN_VALUE(self, state, inst):
        state.append(inst, retval=state.pop(), castval=state.make_temp())
        state.terminate()
    if PYVERSION in ((3, 12),):

        def op_RETURN_CONST(self, state, inst):
            res = state.make_temp('const')
            state.append(inst, retval=res, castval=state.make_temp())
            state.terminate()
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def op_YIELD_VALUE(self, state, inst):
        val = state.pop()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.push(res)
    if PYVERSION in ((3, 11), (3, 12)):

        def op_RAISE_VARARGS(self, state, inst):
            if inst.arg == 0:
                exc = None
                if state.has_active_try():
                    raise UnsupportedError('The re-raising of an exception is not yet supported.', loc=self.get_debug_loc(inst.lineno))
            elif inst.arg == 1:
                exc = state.pop()
            else:
                raise ValueError('Multiple argument raise is not supported.')
            state.append(inst, exc=exc)
            if state.has_active_try():
                self._adjust_except_stack(state)
            else:
                state.terminate()
    elif PYVERSION in ((3, 9), (3, 10)):

        def op_RAISE_VARARGS(self, state, inst):
            in_exc_block = any([state.get_top_block('EXCEPT') is not None, state.get_top_block('FINALLY') is not None])
            if inst.arg == 0:
                exc = None
                if in_exc_block:
                    raise UnsupportedError('The re-raising of an exception is not yet supported.', loc=self.get_debug_loc(inst.lineno))
            elif inst.arg == 1:
                exc = state.pop()
            else:
                raise ValueError('Multiple argument raise is not supported.')
            state.append(inst, exc=exc)
            state.terminate()
    else:
        raise NotImplementedError(PYVERSION)

    def op_BEGIN_FINALLY(self, state, inst):
        temps = []
        for i in range(_EXCEPT_STACK_OFFSET):
            tmp = state.make_temp()
            temps.append(tmp)
            state.push(tmp)
        state.append(inst, temps=temps)

    def op_END_FINALLY(self, state, inst):
        blk = state.pop_block()
        state.reset_stack(blk['entry_stack'])
    if PYVERSION in ((3, 12),):

        def op_END_FOR(self, state, inst):
            state.pop()
            state.pop()
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def op_POP_FINALLY(self, state, inst):
        if inst.arg != 0:
            msg = 'Unsupported use of a bytecode related to try..finally or a with-context'
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def op_CALL_FINALLY(self, state, inst):
        pass

    def op_WITH_EXCEPT_START(self, state, inst):
        state.terminate()

    def op_WITH_CLEANUP_START(self, state, inst):
        state.append(inst)

    def op_WITH_CLEANUP_FINISH(self, state, inst):
        state.append(inst)

    def op_SETUP_LOOP(self, state, inst):
        state.push_block(state.make_block(kind='LOOP', end=inst.get_jump_target()))

    def op_BEFORE_WITH(self, state, inst):
        cm = state.pop()
        yielded = state.make_temp()
        exitfn = state.make_temp(prefix='setup_with_exitfn')
        state.push(exitfn)
        state.push(yielded)
        bc = state._bytecode
        ehhead = bc.find_exception_entry(inst.next)
        ehrelated = [ehhead]
        for eh in bc.exception_entries:
            if eh.target == ehhead.target:
                ehrelated.append(eh)
        end = max((eh.end for eh in ehrelated))
        state.append(inst, contextmanager=cm, exitfn=exitfn, end=end)
        state.push_block(state.make_block(kind='WITH', end=end))
        state.fork(pc=inst.next)

    def op_SETUP_WITH(self, state, inst):
        cm = state.pop()
        yielded = state.make_temp()
        exitfn = state.make_temp(prefix='setup_with_exitfn')
        state.append(inst, contextmanager=cm, exitfn=exitfn)
        if PYVERSION < (3, 9):
            state.push_block(state.make_block(kind='WITH_FINALLY', end=inst.get_jump_target()))
        state.push(exitfn)
        state.push(yielded)
        state.push_block(state.make_block(kind='WITH', end=inst.get_jump_target()))
        state.fork(pc=inst.next)

    def _setup_try(self, kind, state, next, end):
        handler_block = state.make_block(kind=kind, end=None, reset_stack=False)
        state.fork(pc=next, extra_block=state.make_block(kind='TRY', end=end, reset_stack=False, handler=handler_block))

    def op_PUSH_EXC_INFO(self, state, inst):
        tos = state.pop()
        state.push(state.make_temp('exception'))
        state.push(tos)

    def op_SETUP_FINALLY(self, state, inst):
        state.append(inst)
        self._setup_try('FINALLY', state, next=inst.next, end=inst.get_jump_target())
    if PYVERSION in ((3, 11), (3, 12)):

        def op_POP_EXCEPT(self, state, inst):
            state.pop()
    elif PYVERSION in ((3, 9), (3, 10)):

        def op_POP_EXCEPT(self, state, inst):
            blk = state.pop_block()
            if blk['kind'] not in {BlockKind('EXCEPT'), BlockKind('FINALLY')}:
                raise UnsupportedError(f'POP_EXCEPT got an unexpected block: {blk['kind']}', loc=self.get_debug_loc(inst.lineno))
            state.pop()
            state.pop()
            state.pop()
            state.fork(pc=inst.next)
    else:
        raise NotImplementedError(PYVERSION)

    def op_POP_BLOCK(self, state, inst):
        blk = state.pop_block()
        if blk['kind'] == BlockKind('TRY'):
            state.append(inst, kind='try')
        elif blk['kind'] == BlockKind('WITH'):
            state.append(inst, kind='with')
        state.fork(pc=inst.next)

    def op_BINARY_SUBSCR(self, state, inst):
        index = state.pop()
        target = state.pop()
        res = state.make_temp()
        state.append(inst, index=index, target=target, res=res)
        state.push(res)

    def op_STORE_SUBSCR(self, state, inst):
        index = state.pop()
        target = state.pop()
        value = state.pop()
        state.append(inst, target=target, index=index, value=value)

    def op_DELETE_SUBSCR(self, state, inst):
        index = state.pop()
        target = state.pop()
        state.append(inst, target=target, index=index)

    def op_CALL(self, state, inst):
        narg = inst.arg
        args = list(reversed([state.pop() for _ in range(narg)]))
        callable_or_firstarg = state.pop()
        null_or_callable = state.pop()
        if _is_null_temp_reg(null_or_callable):
            callable = callable_or_firstarg
        else:
            callable = null_or_callable
            args = [callable_or_firstarg, *args]
        res = state.make_temp()
        kw_names = state.pop_kw_names()
        state.append(inst, func=callable, args=args, kw_names=kw_names, res=res)
        state.push(res)

    def op_KW_NAMES(self, state, inst):
        state.set_kw_names(inst.arg)

    def op_CALL_FUNCTION(self, state, inst):
        narg = inst.arg
        args = list(reversed([state.pop() for _ in range(narg)]))
        func = state.pop()
        res = state.make_temp()
        state.append(inst, func=func, args=args, res=res)
        state.push(res)

    def op_CALL_FUNCTION_KW(self, state, inst):
        narg = inst.arg
        names = state.pop()
        args = list(reversed([state.pop() for _ in range(narg)]))
        func = state.pop()
        res = state.make_temp()
        state.append(inst, func=func, args=args, names=names, res=res)
        state.push(res)

    def op_CALL_FUNCTION_EX(self, state, inst):
        if inst.arg & 1 and PYVERSION < (3, 10):
            errmsg = 'CALL_FUNCTION_EX with **kwargs not supported'
            raise UnsupportedError(errmsg)
        if inst.arg & 1:
            varkwarg = state.pop()
        else:
            varkwarg = None
        vararg = state.pop()
        func = state.pop()
        if PYVERSION in ((3, 11), (3, 12)):
            if _is_null_temp_reg(state.peek(1)):
                state.pop()
        elif PYVERSION in ((3, 9), (3, 10)):
            pass
        else:
            raise NotImplementedError(PYVERSION)
        res = state.make_temp()
        state.append(inst, func=func, vararg=vararg, varkwarg=varkwarg, res=res)
        state.push(res)

    def _dup_topx(self, state, inst, count):
        orig = [state.pop() for _ in range(count)]
        orig.reverse()
        duped = [state.make_temp() for _ in range(count)]
        state.append(inst, orig=orig, duped=duped)
        for val in orig:
            state.push(val)
        for val in duped:
            state.push(val)
    if PYVERSION in ((3, 12),):

        def op_CALL_INTRINSIC_1(self, state, inst):
            try:
                operand = CALL_INTRINSIC_1_Operand(inst.arg)
            except TypeError:
                raise NotImplementedError(f'op_CALL_INTRINSIC_1({inst.arg})')
            if operand == ci1op.INTRINSIC_STOPITERATION_ERROR:
                state.append(inst, operand=operand)
                state.terminate()
                return
            elif operand == ci1op.UNARY_POSITIVE:
                val = state.pop()
                res = state.make_temp()
                state.append(inst, operand=operand, value=val, res=res)
                state.push(res)
                return
            elif operand == ci1op.INTRINSIC_LIST_TO_TUPLE:
                tos = state.pop()
                res = state.make_temp()
                state.append(inst, operand=operand, const_list=tos, res=res)
                state.push(res)
                return
            else:
                raise NotImplementedError(operand)
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def op_DUP_TOPX(self, state, inst):
        count = inst.arg
        assert 1 <= count <= 5, 'Invalid DUP_TOPX count'
        self._dup_topx(state, inst, count)

    def op_DUP_TOP(self, state, inst):
        self._dup_topx(state, inst, count=1)

    def op_DUP_TOP_TWO(self, state, inst):
        self._dup_topx(state, inst, count=2)

    def op_COPY(self, state, inst):
        state.push(state.peek(inst.arg))

    def op_SWAP(self, state, inst):
        state.swap(inst.arg)

    def op_ROT_TWO(self, state, inst):
        first = state.pop()
        second = state.pop()
        state.push(first)
        state.push(second)

    def op_ROT_THREE(self, state, inst):
        first = state.pop()
        second = state.pop()
        third = state.pop()
        state.push(first)
        state.push(third)
        state.push(second)

    def op_ROT_FOUR(self, state, inst):
        first = state.pop()
        second = state.pop()
        third = state.pop()
        forth = state.pop()
        state.push(first)
        state.push(forth)
        state.push(third)
        state.push(second)

    def op_UNPACK_SEQUENCE(self, state, inst):
        count = inst.arg
        iterable = state.pop()
        stores = [state.make_temp() for _ in range(count)]
        tupleobj = state.make_temp()
        state.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        for st in reversed(stores):
            state.push(st)

    def op_BUILD_TUPLE(self, state, inst):
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        tup = state.make_temp()
        state.append(inst, items=items, res=tup)
        state.push(tup)

    def _build_tuple_unpack(self, state, inst):
        tuples = list(reversed([state.pop() for _ in range(inst.arg)]))
        temps = [state.make_temp() for _ in range(len(tuples) - 1)]
        is_assign = len(tuples) == 1
        if is_assign:
            temps = [state.make_temp()]
        state.append(inst, tuples=tuples, temps=temps, is_assign=is_assign)
        state.push(temps[-1])

    def op_BUILD_TUPLE_UNPACK_WITH_CALL(self, state, inst):
        self._build_tuple_unpack(state, inst)

    def op_BUILD_TUPLE_UNPACK(self, state, inst):
        self._build_tuple_unpack(state, inst)

    def op_LIST_TO_TUPLE(self, state, inst):
        tos = state.pop()
        res = state.make_temp()
        state.append(inst, const_list=tos, res=res)
        state.push(res)

    def op_BUILD_CONST_KEY_MAP(self, state, inst):
        keys = state.pop()
        vals = list(reversed([state.pop() for _ in range(inst.arg)]))
        keytmps = [state.make_temp() for _ in range(inst.arg)]
        res = state.make_temp()
        state.append(inst, keys=keys, keytmps=keytmps, values=vals, res=res)
        state.push(res)

    def op_BUILD_LIST(self, state, inst):
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        lst = state.make_temp()
        state.append(inst, items=items, res=lst)
        state.push(lst)

    def op_LIST_APPEND(self, state, inst):
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        appendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, appendvar=appendvar, res=res)

    def op_LIST_EXTEND(self, state, inst):
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        extendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, extendvar=extendvar, res=res)

    def op_BUILD_MAP(self, state, inst):
        dct = state.make_temp()
        count = inst.arg
        items = []
        for i in range(count):
            v, k = (state.pop(), state.pop())
            items.append((k, v))
        state.append(inst, items=items[::-1], size=count, res=dct)
        state.push(dct)

    def op_MAP_ADD(self, state, inst):
        TOS = state.pop()
        TOS1 = state.pop()
        key, value = (TOS1, TOS)
        index = inst.arg
        target = state.peek(index)
        setitemvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, key=key, value=value, setitemvar=setitemvar, res=res)

    def op_BUILD_SET(self, state, inst):
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        res = state.make_temp()
        state.append(inst, items=items, res=res)
        state.push(res)

    def op_SET_UPDATE(self, state, inst):
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        updatevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, updatevar=updatevar, res=res)

    def op_DICT_UPDATE(self, state, inst):
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        updatevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, updatevar=updatevar, res=res)

    def op_GET_ITER(self, state, inst):
        value = state.pop()
        res = state.make_temp()
        state.append(inst, value=value, res=res)
        state.push(res)

    def op_FOR_ITER(self, state, inst):
        iterator = state.get_tos()
        pair = state.make_temp()
        indval = state.make_temp()
        pred = state.make_temp()
        state.append(inst, iterator=iterator, pair=pair, indval=indval, pred=pred)
        state.push(indval)
        end = inst.get_jump_target()
        if PYVERSION in ((3, 12),):
            state.fork(pc=end)
        elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
            state.fork(pc=end, npop=2)
        else:
            raise NotImplementedError(PYVERSION)
        state.fork(pc=inst.next)

    def op_GEN_START(self, state, inst):
        """Pops TOS. If TOS was not None, raises an exception. The kind
        operand corresponds to the type of generator or coroutine and
        determines the error message. The legal kinds are 0 for generator,
        1 for coroutine, and 2 for async generator.

        New in version 3.10.
        """
        pass

    def op_BINARY_OP(self, state, inst):
        op = dis._nb_ops[inst.arg][1]
        rhs = state.pop()
        lhs = state.pop()
        op_name = ALL_BINOPS_TO_OPERATORS[op].__name__
        res = state.make_temp(prefix=f'binop_{op_name}')
        state.append(inst, op=op, lhs=lhs, rhs=rhs, res=res)
        state.push(res)

    def _unaryop(self, state, inst):
        val = state.pop()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.push(res)
    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_INVERT = _unaryop

    def _binaryop(self, state, inst):
        rhs = state.pop()
        lhs = state.pop()
        res = state.make_temp()
        state.append(inst, lhs=lhs, rhs=rhs, res=res)
        state.push(res)
    op_COMPARE_OP = _binaryop
    op_IS_OP = _binaryop
    op_CONTAINS_OP = _binaryop
    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop
    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop
    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop
    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop

    def op_MAKE_FUNCTION(self, state, inst, MAKE_CLOSURE=False):
        if PYVERSION in ((3, 11), (3, 12)):
            name = None
        elif PYVERSION in ((3, 9), (3, 10)):
            name = state.pop()
        else:
            raise NotImplementedError(PYVERSION)
        code = state.pop()
        closure = annotations = kwdefaults = defaults = None
        if inst.arg & 8:
            closure = state.pop()
        if inst.arg & 4:
            annotations = state.pop()
        if inst.arg & 2:
            kwdefaults = state.pop()
        if inst.arg & 1:
            defaults = state.pop()
        res = state.make_temp()
        state.append(inst, name=name, code=code, closure=closure, annotations=annotations, kwdefaults=kwdefaults, defaults=defaults, res=res)
        state.push(res)

    def op_MAKE_CLOSURE(self, state, inst):
        self.op_MAKE_FUNCTION(state, inst, MAKE_CLOSURE=True)

    def op_LOAD_CLOSURE(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_ASSERTION_ERROR(self, state, inst):
        res = state.make_temp('assertion_error')
        state.append(inst, res=res)
        state.push(res)

    def op_CHECK_EXC_MATCH(self, state, inst):
        pred = state.make_temp('predicate')
        tos = state.pop()
        tos1 = state.get_tos()
        state.append(inst, pred=pred, tos=tos, tos1=tos1)
        state.push(pred)

    def op_JUMP_IF_NOT_EXC_MATCH(self, state, inst):
        pred = state.make_temp('predicate')
        tos = state.pop()
        tos1 = state.pop()
        state.append(inst, pred=pred, tos=tos, tos1=tos1)
        state.fork(pc=inst.next)
        state.fork(pc=inst.get_jump_target())
    if PYVERSION in ((3, 11), (3, 12)):

        def op_RERAISE(self, state, inst):
            exc = state.pop()
            if inst.arg != 0:
                state.pop()
            state.append(inst, exc=exc)
            if state.has_active_try():
                self._adjust_except_stack(state)
            else:
                state.terminate()
    elif PYVERSION in ((3, 9), (3, 10)):

        def op_RERAISE(self, state, inst):
            exc = state.pop()
            state.append(inst, exc=exc)
            state.terminate()
    else:
        raise NotImplementedError(PYVERSION)
    if PYVERSION in ((3, 12),):
        pass
    elif PYVERSION in ((3, 11),):

        def op_LOAD_METHOD(self, state, inst):
            item = state.pop()
            extra = state.make_null()
            state.push(extra)
            res = state.make_temp()
            state.append(inst, item=item, res=res)
            state.push(res)
    elif PYVERSION in ((3, 9), (3, 10)):

        def op_LOAD_METHOD(self, state, inst):
            self.op_LOAD_ATTR(state, inst)
    else:
        raise NotImplementedError(PYVERSION)

    def op_CALL_METHOD(self, state, inst):
        self.op_CALL_FUNCTION(state, inst)