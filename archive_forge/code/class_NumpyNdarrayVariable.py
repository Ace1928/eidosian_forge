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
class NumpyNdarrayVariable(TensorVariable):
    """
    Represents an np.ndarray, but backed by torch Tensor via torch._numpy.ndarray.
    Use this for Tensor.numpy() call.
    """

    @staticmethod
    def create(tx, proxy, **options):
        from .builder import wrap_fx_proxy_cls
        return wrap_fx_proxy_cls(target_cls=NumpyNdarrayVariable, tx=tx, proxy=proxy, **options)

    def var_getattr(self, tx, name):
        from ..utils import numpy_attr_wrapper
        from .builder import wrap_fx_proxy
        result = None
        example_value = self.as_proxy().node.meta['example_value']
        example_ndarray = tnp.ndarray(example_value)

        def insert_into_graph():
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_function', numpy_attr_wrapper, (self.as_proxy(), name), {}))
        if name in ['T', 'real', 'imag']:
            proxy = tx.output.create_proxy('call_function', numpy_attr_wrapper, (self.as_proxy(), name), {})
            result = NumpyNdarrayVariable.create(tx, proxy)
        elif name in ('ndim', 'itemsize'):
            return ConstantVariable.create(getattr(example_ndarray, name))
        elif name in ('shape', 'stride'):
            if not has_free_symbols((r := getattr(example_ndarray, name))):
                return ConstantVariable.create(tuple((int(r) for r in r)))
            return insert_into_graph()
        elif name == 'size':
            if not has_free_symbols((r := example_ndarray.size)):
                return ConstantVariable.create(int(r))
            return insert_into_graph()
        elif name in ['base', 'flags', 'dtype']:
            unimplemented(f'TODO: add support for ndarray.{name}')
        elif name in ['__version__']:
            unimplemented('delegate np.__version__ to NumPy')
        if result is None:
            raise NotImplementedError()
        return result

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from ..utils import numpy_method_wrapper
        if name in ['__len__', 'size', 'tolist']:
            return super().call_method(tx, name, args, kwargs)
        if name == 'tobytes':
            unimplemented('tobytes is not modelled in torch._numpy')
        proxy = tx.output.create_proxy('call_function', numpy_method_wrapper(name), *proxy_args_kwargs([self] + list(args), kwargs))
        return NumpyNdarrayVariable.create(tx, proxy)

    def python_type(self):
        return np.ndarray