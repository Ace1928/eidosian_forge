from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class _UnicodeCompiler(_Compiler):
    """Returns a unicode pluralization rule again."""
    compile_is = _binary_compiler('%s is %s')
    compile_isnot = _binary_compiler('%s is not %s')
    compile_and = _binary_compiler('%s and %s')
    compile_or = _binary_compiler('%s or %s')
    compile_mod = _binary_compiler('%s mod %s')

    def compile_not(self, relation):
        return self.compile_relation(*relation[1], negated=True)

    def compile_relation(self, method, expr, range_list, negated=False):
        ranges = []
        for item in range_list[1]:
            if item[0] == item[1]:
                ranges.append(self.compile(item[0]))
            else:
                ranges.append(f'{self.compile(item[0])}..{self.compile(item[1])}')
        return f'{self.compile(expr)}{(' not' if negated else '')} {method} {','.join(ranges)}'