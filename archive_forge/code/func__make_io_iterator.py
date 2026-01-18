from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
def _make_io_iterator(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXDataIterGetIterInfo(handle, ctypes.byref(name), ctypes.byref(desc), ctypes.byref(num_args), ctypes.byref(arg_names), ctypes.byref(arg_types), ctypes.byref(arg_descs)))
    iter_name = py_str(name.value)
    narg = int(num_args.value)
    param_str = _build_param_doc([py_str(arg_names[i]) for i in range(narg)], [py_str(arg_types[i]) for i in range(narg)], [py_str(arg_descs[i]) for i in range(narg)])
    doc_str = '%s\n\n' + '%s\n' + 'Returns\n' + '-------\n' + 'MXDataIter\n' + '    The result iterator.'
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting data iterator.

        Returns
        -------
        dataiter: Dataiter
            The resulting data iterator.
        """
        param_keys = []
        param_vals = []
        for k, val in kwargs.items():
            param_keys.append(k)
            param_vals.append(str(val))
        param_keys = c_str_array(param_keys)
        param_vals = c_str_array(param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(handle, mx_uint(len(param_keys)), param_keys, param_vals, ctypes.byref(iter_handle)))
        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)
        return MXDataIter(iter_handle, **kwargs)
    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator