from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class Expr(Inst):
    """
    An IR expression (an instruction which can only be part of a larger
    statement).
    """

    def __init__(self, op, loc, **kws):
        assert isinstance(op, str)
        assert isinstance(loc, Loc)
        self.op = op
        self.loc = loc
        self._kws = kws

    def __getattr__(self, name):
        if name.startswith('_'):
            return Inst.__getattr__(self, name)
        return self._kws[name]

    def __setattr__(self, name, value):
        if name in ('op', 'loc', '_kws'):
            self.__dict__[name] = value
        else:
            self._kws[name] = value

    @classmethod
    def binop(cls, fn, lhs, rhs, loc):
        assert isinstance(fn, BuiltinFunctionType)
        assert isinstance(lhs, Var)
        assert isinstance(rhs, Var)
        assert isinstance(loc, Loc)
        op = 'binop'
        return cls(op=op, loc=loc, fn=fn, lhs=lhs, rhs=rhs, static_lhs=UNDEFINED, static_rhs=UNDEFINED)

    @classmethod
    def inplace_binop(cls, fn, immutable_fn, lhs, rhs, loc):
        assert isinstance(fn, BuiltinFunctionType)
        assert isinstance(immutable_fn, BuiltinFunctionType)
        assert isinstance(lhs, Var)
        assert isinstance(rhs, Var)
        assert isinstance(loc, Loc)
        op = 'inplace_binop'
        return cls(op=op, loc=loc, fn=fn, immutable_fn=immutable_fn, lhs=lhs, rhs=rhs, static_lhs=UNDEFINED, static_rhs=UNDEFINED)

    @classmethod
    def unary(cls, fn, value, loc):
        assert isinstance(value, (str, Var, FunctionType))
        assert isinstance(loc, Loc)
        op = 'unary'
        fn = UNARY_BUITINS_TO_OPERATORS.get(fn, fn)
        return cls(op=op, loc=loc, fn=fn, value=value)

    @classmethod
    def call(cls, func, args, kws, loc, vararg=None, varkwarg=None, target=None):
        assert isinstance(func, Var)
        assert isinstance(loc, Loc)
        op = 'call'
        return cls(op=op, loc=loc, func=func, args=args, kws=kws, vararg=vararg, varkwarg=varkwarg, target=target)

    @classmethod
    def build_tuple(cls, items, loc):
        assert isinstance(loc, Loc)
        op = 'build_tuple'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_list(cls, items, loc):
        assert isinstance(loc, Loc)
        op = 'build_list'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_set(cls, items, loc):
        assert isinstance(loc, Loc)
        op = 'build_set'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_map(cls, items, size, literal_value, value_indexes, loc):
        assert isinstance(loc, Loc)
        op = 'build_map'
        return cls(op=op, loc=loc, items=items, size=size, literal_value=literal_value, value_indexes=value_indexes)

    @classmethod
    def pair_first(cls, value, loc):
        assert isinstance(value, Var)
        op = 'pair_first'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def pair_second(cls, value, loc):
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        op = 'pair_second'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def getiter(cls, value, loc):
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        op = 'getiter'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def iternext(cls, value, loc):
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        op = 'iternext'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def exhaust_iter(cls, value, count, loc):
        assert isinstance(value, Var)
        assert isinstance(count, int)
        assert isinstance(loc, Loc)
        op = 'exhaust_iter'
        return cls(op=op, loc=loc, value=value, count=count)

    @classmethod
    def getattr(cls, value, attr, loc):
        assert isinstance(value, Var)
        assert isinstance(attr, str)
        assert isinstance(loc, Loc)
        op = 'getattr'
        return cls(op=op, loc=loc, value=value, attr=attr)

    @classmethod
    def getitem(cls, value, index, loc):
        assert isinstance(value, Var)
        assert isinstance(index, Var)
        assert isinstance(loc, Loc)
        op = 'getitem'
        fn = operator.getitem
        return cls(op=op, loc=loc, value=value, index=index, fn=fn)

    @classmethod
    def typed_getitem(cls, value, dtype, index, loc):
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        op = 'typed_getitem'
        return cls(op=op, loc=loc, value=value, dtype=dtype, index=index)

    @classmethod
    def static_getitem(cls, value, index, index_var, loc):
        assert isinstance(value, Var)
        assert index_var is None or isinstance(index_var, Var)
        assert isinstance(loc, Loc)
        op = 'static_getitem'
        fn = operator.getitem
        return cls(op=op, loc=loc, value=value, index=index, index_var=index_var, fn=fn)

    @classmethod
    def cast(cls, value, loc):
        """
        A node for implicit casting at the return statement
        """
        assert isinstance(value, Var)
        assert isinstance(loc, Loc)
        op = 'cast'
        return cls(op=op, value=value, loc=loc)

    @classmethod
    def phi(cls, loc):
        """Phi node
        """
        assert isinstance(loc, Loc)
        return cls(op='phi', incoming_values=[], incoming_blocks=[], loc=loc)

    @classmethod
    def make_function(cls, name, code, closure, defaults, loc):
        """
        A node for making a function object.
        """
        assert isinstance(loc, Loc)
        op = 'make_function'
        return cls(op=op, name=name, code=code, closure=closure, defaults=defaults, loc=loc)

    @classmethod
    def null(cls, loc):
        """
        A node for null value.

        This node is not handled by type inference. It is only added by
        post-typing passes.
        """
        assert isinstance(loc, Loc)
        op = 'null'
        return cls(op=op, loc=loc)

    @classmethod
    def undef(cls, loc):
        """
        A node for undefined value specifically from LOAD_FAST_AND_CLEAR opcode.
        """
        assert isinstance(loc, Loc)
        op = 'undef'
        return cls(op=op, loc=loc)

    @classmethod
    def dummy(cls, op, info, loc):
        """
        A node for a dummy value.

        This node is a place holder for carrying information through to a point
        where it is rewritten into something valid. This node is not handled
        by type inference or lowering. It's presence outside of the interpreter
        renders IR as illegal.
        """
        assert isinstance(loc, Loc)
        assert isinstance(op, str)
        return cls(op=op, info=info, loc=loc)

    def __repr__(self):
        if self.op == 'call':
            args = ', '.join((str(a) for a in self.args))
            pres_order = self._kws.items() if config.DIFF_IR == 0 else sorted(self._kws.items())
            kws = ', '.join(('%s=%s' % (k, v) for k, v in pres_order))
            vararg = '*%s' % (self.vararg,) if self.vararg is not None else ''
            arglist = ', '.join(filter(None, [args, vararg, kws]))
            return 'call %s(%s)' % (self.func, arglist)
        elif self.op == 'binop':
            lhs, rhs = (self.lhs, self.rhs)
            if self.fn == operator.contains:
                lhs, rhs = (rhs, lhs)
            fn = OPERATORS_TO_BUILTINS.get(self.fn, self.fn)
            return '%s %s %s' % (lhs, fn, rhs)
        else:
            pres_order = self._kws.items() if config.DIFF_IR == 0 else sorted(self._kws.items())
            args = ('%s=%s' % (k, v) for k, v in pres_order)
            return '%s(%s)' % (self.op, ', '.join(args))

    def list_vars(self):
        return self._rec_list_vars(self._kws)

    def infer_constant(self):
        raise ConstantInferenceError('%s' % self, loc=self.loc)