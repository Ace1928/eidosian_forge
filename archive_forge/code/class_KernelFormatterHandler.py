from __future__ import annotations
import itertools
from contextlib import contextmanager
from itertools import chain
from threading import local
from typing import Any, Callable, TYPE_CHECKING, Union
from unittest.mock import patch
import sympy
from torch._inductor.utils import IndentedBuffer
from torch.fx.graph import inplace_methods, magic_methods
from .utils import reduction_num_outputs, sympy_str, sympy_symbol
class KernelFormatterHandler:

    def __init__(self, parent_handler):
        self.parent_handler = parent_handler
        self.output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        from .ir import FlexibleLayout
        args = [index, rindex] if rindex is not None else [index]
        names = ['index', 'rindex'] if rindex is not None else ['index']
        formatter = KernelFormatterHandler(MockHandler())
        with formatter.output.indent(-1):
            formatter.output.writeline(f'def inner_fn({', '.join(names)}):')
        for name, arg in zip(names, args):
            if arg:
                lhs = ', '.join([str('_' if isinstance(v, (int, sympy.Integer)) else v) for v in arg])
                formatter.output.writeline(f'{lhs} = {name}')
        with V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
            result = ir_fn(*args)
            return formatter.getvalue(result)

    def __getattr__(self, name) -> Callable[..., str]:

        def inner(*args, **kwargs):
            line = getattr(self.parent_handler, name)(*args, **kwargs)
            if name == 'indirect_indexing':
                return line
            varname = f'tmp{next(self.var_counter)}'
            self.output.writeline(f'{varname} = {line}')
            return varname
        return inner

    def reduction(self, dtype, src_dtype, reduction_type, value) -> Union[tuple[str, ...], str]:
        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f'tmp{next(self.var_counter)}' for _ in range(num_values)]
        self.output.writeline(f'{','.join(varnames)} = {line}')
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        self.output.writeline(f'return {result}')
        return self.output.getvalue()