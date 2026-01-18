import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array, c_str, mx_uint
from ..base import SymbolHandle
from ..base import check_call
def _symbol_creator(handle, args, kwargs, keys, vals, name, is_np_op, output_is_list=False):
    sym_handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateAtomicSymbol(ctypes.c_void_p(handle), mx_uint(len(keys)), c_str_array(keys), c_str_array([str(v) for v in vals]), ctypes.byref(sym_handle)))
    if args and kwargs:
        raise TypeError('Operators with variable length input can only accept inputSymbols either as positional or keyword arguments, not both')
    create_symbol_fn = _np_symbol_cls if is_np_op else _symbol_cls
    s = create_symbol_fn(sym_handle)
    if args:
        s._compose(*args, name=name)
    elif kwargs:
        s._compose(name=name, **kwargs)
    else:
        s._compose(name=name)
    if is_np_op:
        if s.num_outputs > 1:
            return list(s)
        elif output_is_list:
            return [s]
    return s