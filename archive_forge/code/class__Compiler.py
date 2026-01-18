from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class _Compiler:
    """The compilers are able to transform the expressions into multiple
    output formats.
    """

    def compile(self, arg):
        op, args = arg
        return getattr(self, f'compile_{op}')(*args)
    compile_n = lambda x: 'n'
    compile_i = lambda x: 'i'
    compile_v = lambda x: 'v'
    compile_w = lambda x: 'w'
    compile_f = lambda x: 'f'
    compile_t = lambda x: 't'
    compile_c = lambda x: 'c'
    compile_e = lambda x: 'e'
    compile_value = lambda x, v: str(v)
    compile_and = _binary_compiler('(%s && %s)')
    compile_or = _binary_compiler('(%s || %s)')
    compile_not = _unary_compiler('(!%s)')
    compile_mod = _binary_compiler('(%s %% %s)')
    compile_is = _binary_compiler('(%s == %s)')
    compile_isnot = _binary_compiler('(%s != %s)')

    def compile_relation(self, method, expr, range_list):
        raise NotImplementedError()