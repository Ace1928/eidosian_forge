import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
@staticmethod
@functools.lru_cache(None)
def _binop_handlers():
    op_handlers = {}
    for op, (magic_method_names, in_place_op) in BuiltinVariable._binops().items():
        op_handlers[op] = []
        op_handlers[in_place_op] = []
        forward_name, reverse_name, inplace_name = magic_method_names

        def user_defined_handler(tx, a, b, options, forward_name=forward_name, reverse_name=reverse_name):
            if isinstance(a, UserDefinedVariable):
                return a.call_method(tx, forward_name, [b], {})
            else:
                return b.call_method(tx, reverse_name, [a], {})
        op_handlers[op].append(((UserDefinedVariable, VariableTracker), user_defined_handler))
        op_handlers[op].append(((VariableTracker, UserDefinedVariable), user_defined_handler))

        def user_defined_inplace_handler(tx, a, b, options, forward_name=inplace_name):
            return a.call_method(tx, forward_name, [b], {})
        op_handlers[in_place_op].append(((UserDefinedVariable, VariableTracker), user_defined_inplace_handler))
        op_handlers[in_place_op].append(((VariableTracker, UserDefinedVariable), user_defined_inplace_handler))

        def dynamic_handler(tx, a, b, options, fn=op):
            from .builder import wrap_fx_proxy
            return wrap_fx_proxy(tx, tx.output.create_proxy('call_function', fn, *proxy_args_kwargs([a, b], {})), **options)
        op_handlers[op].append(((SymNodeVariable, VariableTracker), dynamic_handler))
        op_handlers[op].append(((VariableTracker, SymNodeVariable), dynamic_handler))
        op_handlers[in_place_op].append(((SymNodeVariable, VariableTracker), dynamic_handler))
        op_handlers[in_place_op].append(((VariableTracker, SymNodeVariable), dynamic_handler))

    def tuple_add_handler(tx, a, b, options):
        return TupleVariable(a.items + list(b.unpack_var_sequence(tx)), **options)

    def size_add_handler(tx, a, b, options):
        return SizeVariable(a.items + list(b.unpack_var_sequence(tx)), **options)
    list_like_addition_handlers = [((SizeVariable, SizeVariable), size_add_handler), ((TupleVariable, TupleVariable), tuple_add_handler), ((TupleVariable, ConstantVariable), tuple_add_handler), ((ConstantVariable, TupleVariable), lambda tx, a, b, options: TupleVariable(list(a.unpack_var_sequence(tx)) + b.items, **options)), ((BaseListVariable, BaseListVariable), lambda tx, a, b, options: type(a)(a.items + b.items, **options))]
    op_handlers[operator.add].extend(list_like_addition_handlers)

    def list_iadd_handler(tx, a, b, options):
        if not a.mutable_local or not b.has_unpack_var_sequence(tx):
            return None
        return tx.replace_all(a, ListVariable(list(a.items) + list(b.unpack_var_sequence(tx)), **options))
    list_like_iadd_handlers = [((ListVariable, VariableTracker), list_iadd_handler), ((TupleVariable, TupleVariable), tuple_add_handler), ((TupleVariable, ConstantVariable), tuple_add_handler)]
    op_handlers[operator.iadd].extend(list_like_iadd_handlers)

    def expand_list_like(tx, lst, const, options):
        return lst.__class__(items=lst.items * const.as_python_constant(), mutable_local=MutableLocal(), **options)
    list_like_expansion_handlers = [((ListVariable, ConstantVariable), expand_list_like), ((TupleVariable, ConstantVariable), expand_list_like), ((ConstantVariable, ListVariable), lambda tx, a, b, options: expand_list_like(tx, b, a, options)), ((ConstantVariable, TupleVariable), lambda tx, a, b, options: expand_list_like(tx, b, a, options))]
    op_handlers[operator.mul].extend(list_like_expansion_handlers)
    return op_handlers