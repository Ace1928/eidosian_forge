import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
class SymNodeVariable(VariableTracker):
    """
    Represents a symbolic size, e.g., as returned by tensor.size(0)
    """

    @classmethod
    def create(cls, tx, proxy, sym_num, **options):
        if 'example_value' in proxy.node.meta:
            assert proxy.node.meta['example_value'] == sym_num
        if sym_num is None:
            sym_num = get_fake_value(proxy.node, tx)
        proxy.node.meta['example_value'] = sym_num
        if isinstance(sym_num, (sympy.Integer, int, bool)):
            sym_num = int(sym_num) if isinstance(sym_num, sympy.Integer) else sym_num
            return ConstantVariable.create(sym_num)
        return SymNodeVariable(proxy, sym_num, **options)

    def __init__(self, proxy, sym_num, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.sym_num = sym_num

    def python_type(self):
        if isinstance(self.sym_num, SymTypes):
            return self.sym_num.node.pytype
        else:
            return type(self.sym_num)

    def as_proxy(self):
        return self.proxy

    def evaluate_expr(self, output_graph=None):
        try:
            return guard_scalar(self.sym_num)
        except GuardOnDataDependentSymNode as e:
            raise UserError(UserErrorType.ANTI_PATTERN, f'Consider annotating your code using torch._constrain_as_*(). {str(e)}', case_name='constrain_as_size_example')

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .builder import wrap_fx_proxy
        return wrap_fx_proxy(tx, tx.output.create_proxy('call_method', name, *proxy_args_kwargs([self] + list(args), kwargs)))