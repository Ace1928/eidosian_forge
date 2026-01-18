from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class _JavaScriptCompiler(_GettextCompiler):
    """Compiles the expression to plain of JavaScript."""
    compile_i = lambda x: 'parseInt(n, 10)'
    compile_v = compile_zero
    compile_w = compile_zero
    compile_f = compile_zero
    compile_t = compile_zero

    def compile_relation(self, method, expr, range_list):
        code = _GettextCompiler.compile_relation(self, method, expr, range_list)
        if method == 'in':
            expr = self.compile(expr)
            code = f'(parseInt({expr}, 10) == {expr} && {code})'
        return code