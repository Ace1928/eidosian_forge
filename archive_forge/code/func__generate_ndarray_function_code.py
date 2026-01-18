import os as _os
import ctypes
import numpy as _np  # pylint: disable=unused-import
from ._internal import NDArrayBase, _imperative_invoke # pylint: disable=unused-import
from ..ndarray_doc import _build_doc
from ..base import mx_uint, check_call, _LIB, py_str, _init_op_module, _Null, _is_np_op, _output_is_list  # pylint: disable=unused-import
from ..util import use_np_shape  # pylint: disable=unused-import
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
def _generate_ndarray_function_code(handle, op_name, func_name, signature_only=False):
    """Generate function for ndarray op by handle and function op_name."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()
    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(handle, ctypes.byref(real_name), ctypes.byref(desc), ctypes.byref(num_args), ctypes.byref(arg_names), ctypes.byref(arg_types), ctypes.byref(arg_descs), ctypes.byref(key_var_num_args), ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(op_name, py_str(desc.value), arg_names, arg_types, [py_str(arg_descs[i]) for i in range(narg)], key_var_num_args, ret_type)
    dtype_name = None
    arr_name = None
    ndsignature = []
    signature = []
    ndarg_names = []
    kwarg_names = []
    for i in range(narg):
        name, atype = (arg_names[i], arg_types[i])
        if name == 'dtype':
            dtype_name = name
            signature.append('%s=_Null' % name)
        elif atype.startswith('NDArray') or atype.startswith('Symbol'):
            assert not arr_name, 'Op can only have one argument with variable size and it must be the last argument.'
            if atype.endswith('[]'):
                ndsignature.append('*%s' % name)
                arr_name = name
            else:
                ndsignature.append('%s=None' % name)
                ndarg_names.append(name)
        else:
            signature.append('%s=_Null' % name)
            kwarg_names.append(name)
    signature.append('out=None')
    signature.append('name=None')
    signature.append('**kwargs')
    signature = ndsignature + signature
    code = []
    is_np_op = _is_np_op(op_name)
    output_is_list = _output_is_list(op_name)
    doc_str_idx = 1
    if is_np_op:
        doc_str_idx = 2
    if arr_name:
        code.append('\ndef %s(*%s, **kwargs):' % (func_name, arr_name))
        if not signature_only:
            code.append('\n    ndargs = []\n    for i in {}:\n        assert isinstance(i, NDArrayBase), \\\n            "Positional arguments must have NDArray type, " \\\n            "but got %s"%str(i)\n        ndargs.append(i)'.format(arr_name))
            if dtype_name is not None:
                code.append("\n    if '%s' in kwargs:\n        if _np.dtype(kwargs['%s']).names:\n            kwargs['%s'] = _np.dtype(kwargs['%s']).names[0]\n        else:\n            kwargs['%s'] = _np.dtype(kwargs['%s']).name " % (dtype_name, dtype_name, dtype_name, dtype_name, dtype_name, dtype_name))
            code.append("\n    _ = kwargs.pop('name', None)\n    out = kwargs.pop('out', None)\n    keys = list(kwargs.keys())\n    vals = list(kwargs.values())")
    else:
        code.append('\ndef %s(%s):' % (func_name, ', '.join(signature)))
        if not signature_only:
            code.append('\n    ndargs = []\n    keys = list(kwargs.keys())\n    vals = list(kwargs.values())')
            for name in ndarg_names:
                code.append('\n    if {name} is not None:\n        assert isinstance({name}, NDArrayBase), \\\n            "Argument {name} must have NDArray type, but got %s"%str({name})\n        ndargs.append({name})'.format(name=name))
            for name in kwarg_names:
                code.append("\n    if %s is not _Null:\n        keys.append('%s')\n        vals.append(%s)" % (name, name, name))
            if dtype_name is not None:
                if is_np_op:
                    code.append("\n    if %s is not _Null and %s is not None:\n        keys.append('%s')\n        vals.append(_np.dtype(%s).name)" % (dtype_name, dtype_name, dtype_name, dtype_name))
                else:
                    code.append("\n    if %s is not _Null:\n        keys.append('%s')\n        if _np.dtype(%s).names:\n            vals.append(_np.dtype(%s).names[0])\n        else:\n            vals.append(_np.dtype(%s).name) " % (dtype_name, dtype_name, dtype_name, dtype_name, dtype_name))
    verify_ndarrays_fn = _verify_all_np_ndarrays.__name__ if is_np_op else _verify_all_legacy_ndarrays.__name__
    if not signature_only:
        code.append('\n    {verify_fn}("{op_name}", "{func_name}", ndargs, out)\n        '.format(verify_fn=verify_ndarrays_fn, op_name=op_name, func_name=func_name))
        code.append('\n    return _imperative_invoke(%d, ndargs, keys, vals, out, %s, %s)' % (handle.value, str(is_np_op), str(output_is_list)))
    else:
        code.append('\n    return (0,)')
    doc_str_lines = _os.linesep + ''.join(['    ' + s if s.strip() else s for s in 'r"""{doc_str}"""'.format(doc_str=doc_str).splitlines(True)])
    code.insert(doc_str_idx, doc_str_lines)
    return (''.join(code), doc_str)