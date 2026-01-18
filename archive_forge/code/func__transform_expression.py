import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _transform_expression(expression):
    if dimension_name(expression.dimension) in exclude:
        dimension = expression.dimension
    else:
        dimension = self._transform_dimension(kdims, vdims, expression.dimension)
    expression = expression.clone(dimension)
    ops = []
    for op in expression.ops:
        new_op = dict(op)
        new_args = []
        for arg in op['args']:
            if isinstance(arg, dim):
                arg = _transform_expression(arg)
            new_args.append(arg)
        new_op['args'] = tuple(new_args)
        new_kwargs = {}
        for kw, kwarg in op['kwargs'].items():
            if isinstance(kwarg, dim):
                kwarg = _transform_expression(kwarg)
            new_kwargs[kw] = kwarg
        new_op['kwargs'] = new_kwargs
        ops.append(new_op)
    expression.ops = ops
    return expression